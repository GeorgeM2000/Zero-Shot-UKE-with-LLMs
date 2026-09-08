import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Count how many words and non-words exist in the extracted keyphrases list (per dataset)
# Additionally, calculate the average number of words per extracted keyphrase for the entire dataset
def process_keyphrases(perdoc_keyphrases):
    
    # Each element in this list will be the average number of words calculated from all the keywords/keyphrases of a document
    perdoc_avg_no_words = [] 
    
    total_word_count = 0
    total_non_word_count = 0

    # {perdoc_keyphrases} is a 2D list. Each inner list contains keyphrases for one document: [keyphrase 1, ..., keyphrase N]
    for keyphrases_list in perdoc_keyphrases: 

        total_no_words_per_doc = 0

        for keyphrase in keyphrases_list:

            tokens = [token for token in keyphrase.split()]
            
            for t in tokens:
                if t.isalpha(): # Check if it's a valid word
                    total_no_words_per_doc += 1
                    total_word_count += 1
                else:
                    total_non_word_count += 1

        perdoc_avg_no_words.append(total_no_words_per_doc / len(keyphrases_list))
    

    # np.mean(perdoc_avg_no_words) is the average value of all average values that exist in {perdoc_avg_no_words}
    return total_word_count, total_non_word_count, np.mean(perdoc_avg_no_words)



def count_word_overlap_matches(candidate_n_keyphrases, candidate_orig_keyphrases, reference_n_keyphrases, reference_orig_keyphrases, threshold=0.25):
    """
    Count how many candidate keyphrases match the reference keyphrases
    based on word overlap (>= threshold), using OR logic on normalized and original versions.

    A match means: at least `threshold` fraction of words in a candidate
    keyphrase appear in the words of SOME reference keyphrase (normalized OR original).
    """
    matches = 0
    matched_indices = set()  # To avoid matching the same reference keyphrase multiple times

    for cand_kw_n, cand_kw_orig in zip(candidate_n_keyphrases, candidate_orig_keyphrases):
        
        cand_words_n      = cand_kw_n.lower().split() # Normalized words of candidate keyphrase
        cand_words_orig = cand_kw_orig.lower().split() # Original words of candidate keyphrase
        cand_len = len(cand_words_n)

        if cand_len == 0:
            continue

        for idx, (ref_kw_n, ref_kw_orig) in enumerate(zip(reference_n_keyphrases, reference_orig_keyphrases)):
            if idx in matched_indices:
                continue

            ref_words_n      = set(ref_kw_n.lower().split())       # Normalized reference words
            ref_words_orig   = set(ref_kw_orig.lower().split())  # Original reference words

            # OR logic: check overlap on normalized OR original
            overlap_normalized = sum(1 for w in cand_words_n if w in ref_words_n)
            overlap_orig    = sum(1 for w in cand_words_orig if w in ref_words_orig)

            if (overlap_normalized / cand_len >= threshold) or (overlap_orig / cand_len >= threshold):
                matches += 1
                matched_indices.add(idx)
                break  # Stop once we match this candidate to one reference keyphrase

    return matches




# -----------------------------------------------------------------------
# Lemmatization helper (replaces PorterStemmer).
#
# spaCy lemmatization is linguistically correct (dictionary/POS-aware)
# rather than rule-based suffix stripping, so it avoids the occasional
# garbage stems Porter produces, while still collapsing inflected variants
# ("networks" -> "network", "optimizing"/"optimization" -> "optimize")
# onto a shared root for the word-overlap similarity metric to work well.
#
# Reuses the spaCy model already loaded elsewhere in the pipeline
# (spacy_model_path / nlp), so no extra model load cost.
# -----------------------------------------------------------------------
def lemmatize_keyphrases(keyphrases_list, nlp): # keyphrases_list for only one document in a dataset
    """
    Lemmatize a list of keyword/keyphrase strings using spaCy.

    Args:
        keyphrases_list (list of str): Original keyword/keyphrase strings.
        nlp: A loaded spaCy Language object (e.g. spacy.load(spacy_model_path)).

    Returns:
        list of str: Space-joined lemmatized tokens per keyphrase, e.g.
            "neural networks" -> "neural network"
            "optimizing performance" -> "optimize performance"
    """
    # nlp.pipe batches the keyphrases list through spaCy's pipeline efficiently,
    # rather than calling nlp(kw) once per keyphrase in a Python loop.
    lemmatized = []
    for kw in nlp.pipe(keyphrases_list):
        lemmas = [token.lemma_.lower() for token in kw if not token.is_space]
        lemmatized.append(" ".join(lemmas))

    return lemmatized





def cluster_keyphrases(keyphrases, normalized_keyphrases, similarity_threshold=0.25): 
    n = len(keyphrases)
 
    # Edge cases
    if n == 0:
        return []
    if n == 1:
        return list(keyphrases)
 
    # -----------------------------------------------------------------
    # Vectorized similarity matrix.
    # Build a binary bag-of-words matrix (each row = one keyphrase's normalized
    # tokens as a 0/1 vector over the vocabulary), then compute the full
    # pairwise cosine similarity matrix in one C-level operation.
    # This replaces the O(n^2) Python loop calling keyphrase_similarity(),
    # and replaces the redundant per-pair set() construction with a single
    # vectorization pass over all n keyphrases.
    # -----------------------------------------------------------------
    vectorizer = CountVectorizer(binary=True, tokenizer=str.split, token_pattern=None, lowercase=False)
    X = vectorizer.fit_transform(normalized_keyphrases)  # shape (n, vocab_size), sparse binary matrix
 
    # cosine_similarity(overlap / sqrt(len_a * len_b)) on binary vectors is exactly
    # equivalent to: |A ∩ B| / sqrt(|A| * |B|)
    sim_matrix = cosine_similarity(X)  # shape (n, n), dense float array, diagonal = 1.0
 
    # Convert to distance matrix
    dist_matrix = 1 - sim_matrix

    # Numerical safety: cosine_similarity can yield values like 1.0000000002 due to
    # floating point error, which would make distances slightly negative.
    np.clip(dist_matrix, 0, None, out=dist_matrix)
 
    # Clustering with average linkage
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
        compute_full_tree=True,
        n_clusters=None,  # No need to predefine number of clusters
    )
    labels = clustering.fit_predict(dist_matrix)
 
    # -----------------------------------------------------------------
    # Vectorized centroid selection.
    # For each cluster, instead of a Python loop computing np.mean() per
    # member, slice the similarity sub-matrix for the cluster's indices in
    # one shot and compute row-wise average similarity (excluding self-
    # similarity of 1.0) using NumPy array ops.
    # -----------------------------------------------------------------
    centroids = []
    labels = np.asarray(labels)
 
    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0]
 
        if len(indices) == 1:
            centroids.append(keyphrases[indices[0]])
            continue
 
        # Sub-matrix of pairwise similarities within this cluster
        cluster_sims = sim_matrix[np.ix_(indices, indices)]  # shape (k, k)
 
        # Each row sums to (k-1) "real" similarities + 1.0 self-similarity on the diagonal.
        # Subtract the diagonal (always 1.0) and divide by (k-1) to get the average
        # similarity to all *other* members in the cluster.
        k = len(indices)
        row_sums = cluster_sims.sum(axis=1) - np.diag(cluster_sims)
        avg_sims = row_sums / (k - 1)
 
        best_local_idx = np.argmax(avg_sims)
        best_idx = indices[best_local_idx]
        centroids.append(keyphrases[best_idx])
 
    return centroids











def cluster_keyphrase_embeddings(keyphrases, embeddings, similarity_threshold=0.8):
    """
    Cluster keyphrases using Hierarchical Agglomerative Clustering (average linkage)
    based on embedding cosine similarity, and return representative keyphrases.
 
    Args:
        keyphrases (list of str): Candidate keywords/keyphrases.
        embeddings (np.ndarray): Corresponding embedding vectors (n x d).
        similarity_threshold (float): Minimum cosine similarity for clustering.
 
    Returns:
        list of str: Cluster representative keyphrases (centroids).
    """
 
    #keyphrases = list(keyphrases)
    #embeddings = np.asarray(embeddings)  # Avoid a forced copy if already an ndarray
 
    n = len(keyphrases)
 
    # Edge case: empty or single element
    if n == 0:
        return []
    if n == 1:
        return keyphrases
 
    # -----------------------------
    # 1. Compute cosine similarity matrix (already vectorized)
    # -----------------------------
    sim_matrix = cosine_similarity(embeddings)
 
    # -----------------------------
    # 2. Convert similarity -> distance
    # -----------------------------
    dist_matrix = 1 - sim_matrix
    # Numerical safety: cosine_similarity can produce values like 1.0000000002
    # due to floating point error, which would otherwise create tiny negative
    # distances that can upset AgglomerativeClustering.
    np.clip(dist_matrix, 0, None, out=dist_matrix)
 
    # -----------------------------
    # 3. HAC clustering 
    # -----------------------------
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
        n_clusters=None,
        compute_full_tree=True
    )
 
    labels = clustering.fit_predict(dist_matrix)
    labels = np.asarray(labels)
 
    # -----------------------------
    # 4. Select cluster representatives (vectorized)
    #
    # Original code: for each cluster, looped over every member in Python,
    # building a list comprehension of pairwise sims and calling np.mean()
    # per member -- O(k^2) Python-level work per cluster, repeated across
    # all clusters (effectively O(n*k) Python overhead in the worst case).
    #
    # Replacement: slice the similarity sub-matrix for each cluster's
    # indices in one shot via np.ix_, then compute row-wise average
    # similarity (excluding the diagonal self-similarity of 1.0) using
    # vectorized NumPy operations instead of nested Python loops.
    # -----------------------------
    centroids = []
 
    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0]
 
        if len(indices) == 1:
            centroids.append(keyphrases[indices[0]])
            continue
 
        cluster_sims = sim_matrix[np.ix_(indices, indices)]  # shape (k, k)
        k = len(indices)
 
        # Subtract the diagonal (always 1.0, self-similarity) from each row sum,
        # then divide by (k - 1) to get the average similarity to all *other*
        # members of the cluster -- equivalent to the original np.mean(sims).
        row_sums = cluster_sims.sum(axis=1) - np.diag(cluster_sims)
        avg_sims = row_sums / (k - 1)
 
        best_local_idx = np.argmax(avg_sims)
        best_idx = indices[best_local_idx]
        centroids.append(keyphrases[best_idx])
 
    return centroids