import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity



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



# Similarity function: overlap / max(lenA, lenB)
def keyword_similarity(set_a, set_b): # Based on exact similarity 
    overlap = len(set(set_a) & set(set_b))
    denom = max(len(set_a), len(set_b))
    return overlap / denom if denom > 0 else 0.0



def cluster_keywords(keywords, stemmed_keywords, similarity_threshold=0.25):
    """
    Cluster keywords using Hierarchical Agglomerative Clustering (average linkage)
    with word overlap similarity, and return centroid keywords with their scores.

    Args:
        keywords (list of str): Candidate keywords.
        scores (list of float): Scores corresponding to each keyword.
        similarity_threshold (float): Minimum overlap similarity (default=0.25).

    Returns:
        list of (keyword, score): Cluster centroids and their scores.
    """

    # Build similarity matrix
    n = len(keywords)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = keyword_similarity(stemmed_keywords[i], stemmed_keywords[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Convert to distance matrix
    dist_matrix = 1 - sim_matrix

    # Clustering with average linkage
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
        compute_full_tree=True,
        n_clusters=None # There is no need to define the number of clusters
    )
    labels = clustering.fit_predict(dist_matrix)

    # Find cluster centroids
    centroids = []
    for cluster_id in set(labels): # Take all the unique labels, e.g., 0, 1, 2, 3, ...
        indices = [idx for idx, lbl in enumerate(labels) if lbl == cluster_id] # Organize the keyword indices based on their label
        if len(indices) == 1:
            # Single keyword cluster
            idx = indices[0]
            centroids.append(keywords[idx])
        else:
            # Compute centroid: max avg similarity within cluster
            best_idx, best_sim = None, -1
            for idx in indices:
                sims = [sim_matrix[idx, j] for j in indices if j != idx]
                avg_sim = np.mean(sims) if sims else 0
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_idx = idx
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

    keywords = list(keywords)
    embeddings = np.array(embeddings)

    n = len(keywords)

    # Edge case: empty or single element
    if n == 0:
        return []
    if n == 1:
        return keywords

    # -----------------------------
    # 1. Compute cosine similarity matrix
    # -----------------------------
    sim_matrix = cosine_similarity(embeddings)

    # -----------------------------
    # 2. Convert similarity -> distance
    # -----------------------------
    dist_matrix = 1 - sim_matrix

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

    # -----------------------------
    # 4. Select cluster representatives
    # -----------------------------
    centroids = []

    for cluster_id in set(labels):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]

        if len(indices) == 1:
            centroids.append(keywords[indices[0]])
        else:
            best_idx, best_score = None, -1

            for i in indices:
                sims = [sim_matrix[i, j] for j in indices if j != i]
                avg_sim = np.mean(sims) if sims else 0

                if avg_sim > best_score:
                    best_score = avg_sim
                    best_idx = i

            centroids.append(keywords[best_idx])

    return centroids





