import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# This function calculates how many words and non-words exist in the extracted keyphrases per dataset
# Additionally, it calculates the average number of words per extracted keyphrase for the entire dataset
def process_keyphrases(perdoc_keyphrases):
    perdoc_avg_no_words = []
    total_word_count = 0
    total_non_word_count = 0

    for keyphrases_list in perdoc_keyphrases: # perdoc_keyphrases is a 2D list. Each inner list contains keyphrases for one document: [keyphrase 1, ..., keyphrase N]

        total_no_words_per_doc = 0

        for keyphrase in keyphrases_list:

            tokens = [token for token in keyphrase.split()]
            
            for t in tokens:
                if t.isalpha():  # Check if it's a valid word
                    total_no_words_per_doc += 1
                    total_word_count += 1
                else:
                    total_non_word_count += 1

        perdoc_avg_no_words.append(total_no_words_per_doc / len(keyphrases_list))
    
    return total_word_count, total_non_word_count, np.mean(perdoc_avg_no_words)



def count_word_overlap_matches(candidate_keywords, candidate_keywords_orig, reference_keywords, reference_keywords_orig, threshold=0.25):
    """
    Count how many candidate keywords match the reference keywords
    based on word overlap (>= threshold), using OR logic on stemmed and original versions.

    A match means: at least `threshold` fraction of words in a candidate
    keyword appear in the words of SOME reference keyword (stemmed OR original).
    """
    matches = 0
    matched_indices = set()  # To avoid matching the same reference keyword multiple times

    for cand_kw, cand_kw_orig in zip(candidate_keywords, candidate_keywords_orig):
        cand_words      = cand_kw.lower().split()
        cand_words_orig = cand_kw_orig.lower().split()
        cand_len = len(cand_words)
        if cand_len == 0:
            continue

        for idx, (ref_kw, ref_kw_orig) in enumerate(zip(reference_keywords, reference_keywords_orig)):
            if idx in matched_indices:
                continue

            ref_words      = set(ref_kw.lower().split())       # stemmed reference words
            ref_words_orig = set(ref_kw_orig.lower().split())  # original reference words

            # OR logic: check overlap on stemmed OR original
            overlap_stemmed = sum(1 for w in cand_words if w in ref_words)
            overlap_orig    = sum(1 for w in cand_words_orig if w in ref_words_orig)

            if (overlap_stemmed / cand_len >= threshold) or (overlap_orig / cand_len >= threshold):
                matches += 1
                matched_indices.add(idx)
                break  # Stop once we match this candidate to one reference keyword

    return matches


def cluster_keywords(keywords, stemmed_keywords, similarity_threshold=0.25):
    n = len(keywords)
 
    # Edge cases
    if n == 0:
        return []
    if n == 1:
        return list(keywords)
 
    # -----------------------------------------------------------------
    # Vectorized similarity matrix.
    # Build a binary bag-of-words matrix (each row = one keyword's stemmed
    # tokens as a 0/1 vector over the vocabulary), then compute the full
    # pairwise cosine similarity matrix in one C-level operation.
    # This replaces the O(n^2) Python loop calling keyword_similarity(),
    # and replaces the redundant per-pair set() construction with a single
    # vectorization pass over all n keywords.
    # -----------------------------------------------------------------
    vectorizer = CountVectorizer(binary=True, tokenizer=str.split, token_pattern=None, lowercase=False)
    X = vectorizer.fit_transform(stemmed_keywords)  # shape (n, vocab_size), sparse binary matrix
 
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
            centroids.append(keywords[indices[0]])
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
        centroids.append(keywords[best_idx])
 
    return centroids


def cluster_keywords_embeddings(keywords, embeddings, similarity_threshold=0.8):
    """
    Cluster keywords using Hierarchical Agglomerative Clustering (average linkage)
    based on embedding cosine similarity, and return representative keywords.
 
    Args:
        keywords (list of str): Candidate keywords/keyphrases.
        embeddings (np.ndarray): Corresponding embedding vectors (n x d).
        similarity_threshold (float): Minimum cosine similarity for clustering.
 
    Returns:
        list of str: Cluster representative keywords (centroids).
    """
 
    #keywords = list(keywords)
    #embeddings = np.asarray(embeddings)  # avoid a forced copy if already an ndarray
 
    n = len(keywords)
 
    # Edge case: empty or single element
    if n == 0:
        return []
    if n == 1:
        return keywords
 
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
            centroids.append(keywords[indices[0]])
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
        centroids.append(keywords[best_idx])
 
    return centroids