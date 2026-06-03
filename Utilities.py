import numpy as np

from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering



# This function calculates how many words and non-words exist in the keywords list
def process_keywords(keywords_list, output_file=None):
    word_counts = []  
    total_word_count = 0
    total_non_word_count = 0

    for keywords in keywords_list: # keywords_list is a 2D list. Each inner list contains keywords: [keyword 1, ..., keyword N]
        keywords_word_count = 0  
        keywords_non_word_count = 0

        for keyword in keywords: # For each keyword in [keyword 1, ..., keyword N]

            # Split on whitespace or dash but preserve valid words
            words = [word for part in keyword.split('-') for word in part.split()]
            
            for word in words:
                if word.isalpha():  # Check if it's a valid word
                    keywords_word_count += 1
                    total_word_count += 1
                else:
                    keywords_non_word_count += 1
                    total_non_word_count += 1

        word_counts.append(keywords_word_count)
    
    avg_word_count = 0
    for count in word_counts:
        avg_word_count += count

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for count in word_counts:
                f.write(str(count) + '\n')
    
    return total_word_count, total_non_word_count, avg_word_count / len(word_counts)



def count_word_overlap_matches(candidate_keywords, reference_keywords, threshold=0.25):
    """
    Count how many candidate keywords match the reference keywords
    based on word overlap (≥ threshold).
    
    A match means: at least `threshold` fraction of words in a candidate
    keyword appear in the words of SOME reference keyword.
    """
    matches = 0
    matched_indices = set()  # To avoid matching the same reference keyword multiple times

    for cand_kw in candidate_keywords:
        cand_words = cand_kw.lower().split()
        cand_len = len(cand_words)

        if cand_len == 0:
            continue

        for idx, ref_kw in enumerate(reference_keywords):
            if idx in matched_indices:
                continue
            
            ref_words = set(ref_kw.lower().split()) # We use a set for fast lookups
            overlap = sum(1 for w in cand_words if w in ref_words)

            if overlap / cand_len >= threshold:
                matches += 1
                matched_indices.add(idx)
                break  # Stop once we match this candidate to one reference keyword
    
    return matches 



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



def find_knee_with_kneedle(scores):
    """
    Find the knee (elbow) in a list of sorted scores (descending) using Kneedle.
    
    Args:
        scores (list or np.array): Sorted list of scores (high to low).
    
    Returns:
        list: Indices of keywords to keep (up to and including knee).
    """
    scores = np.array(scores)
    x = np.arange(len(scores))

    # Kneedle works best if curve is convex and decreasing
    kneedle = KneeLocator(x, scores, curve="convex", direction="decreasing")
    knee = kneedle.knee

    if knee is None:
        # If no knee detected, keep all
        return list(range(len(scores)))
    else:
        return list(range(knee + 1))
    




# Similarity function: overlap / max(lenA, lenB)
def keyword_similarity(set_a, set_b): # Based on exact similarity 
    overlap = len(set(set_a) & set(set_b))
    denom = max(len(set_a), len(set_b))
    return overlap / denom if denom > 0 else 0.0







def cluster_keywords(keywords, stemmed_keywords, scores, similarity_threshold=0.25):
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
            centroids.append((keywords[idx], scores[idx]))
        else:
            # Compute centroid: max avg similarity within cluster
            best_idx, best_sim = None, -1
            for idx in indices:
                sims = [sim_matrix[idx, j] for j in indices if j != idx]
                avg_sim = np.mean(sims) if sims else 0
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_idx = idx
            centroids.append((keywords[best_idx], scores[best_idx]))

    return centroids








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








def get_top_centroids(extracted_keywords):
    centroid_keywords = []

    for i, ekws in enumerate(extracted_keywords):
        # Separate scores and keywords
        scores, keywords = zip(*[(score, keyword) for score, keyword in ekws])

        # Cluster and sort by score descending
        centroids = sorted(cluster_keywords(list(keywords), list(scores), similarity_threshold=0.25), key=lambda x: x[1], reverse=True)

        # Apply knee method on scores
        knee_indices = find_knee_with_kneedle([score for _, score in centroids])

        # Build best centroids string
        centroid_keywords.append([centroids[idx][0] for idx in knee_indices])
    
    return centroid_keywords, len(knee_indices)