# output_formatter.py

import json
import numpy as np

def load_index_to_word(json_path):
    with open(json_path, "r") as f:
        word_to_index = json.load(f)
    return {v: k for k, v in word_to_index.items()}

def save_cluster_assignments(article_ids, responsibilities, filename="cluster_assignments.txt"):
    cluster_ids = np.argmax(responsibilities, axis=1)
    with open(filename, "w") as f:
        for article_id, cluster_id in zip(article_ids, cluster_ids):
            f.write(f"{article_id}\t{cluster_id}\n")

def save_cluster_stats(means, variances, index_to_word, filename="cluster_stats.txt"):
    with open(filename, "w") as f:
        for k in range(len(means)):
            f.write(f"Cluster {k}:\n")
            top_indices = np.argsort(means[k])[::-1][:5]
            for idx in top_indices:
                word = index_to_word.get(idx, f"[word_{idx}]")
                f.write(f"{word}\tmean={means[k][idx]:.6f}\tvariance={variances[k][idx]:.6f}\n")
            f.write("\n")

def save_em_parameters(weights, means, variances, index_to_word, filename="em_parameters.txt"):
    with open(filename, "w") as f:
        for k in range(len(means)):
            f.write(f"Cluster {k}:\n")
            f.write(f"Weight: {weights[k]:.6f}\n")
            top_indices = np.argsort(means[k])[::-1][:10]
            f.write("Top words (mean, variance):\n")
            for idx in top_indices:
                word = index_to_word.get(idx, f"[word_{idx}]")
                f.write(f"{word}: mean={means[k][idx]:.6f}, var={variances[k][idx]:.6f}\n")
            f.write("\n")

def save_convergence_log(log_likelihoods, filename="convergence_log.txt"):
    with open(filename, "w") as f:
        for i, ll in enumerate(log_likelihoods):
            f.write(f"Iteration {i + 1}: Log-likelihood = {ll:.6f}\n")

def save_ascii_wordclouds(means, index_to_word, filename="ascii_wordclouds.txt"):
    with open(filename, "w") as f:
        for k in range(len(means)):
            f.write(f"Cluster {k} ASCII Word Cloud:\n")
            top_indices = np.argsort(means[k])[::-1][:15]
            for idx in top_indices:
                word = index_to_word.get(idx, f"[word_{idx}]")
                count = int(means[k][idx] * 1000)
                ascii_word = word * max(1, min(count, 20))
                f.write(f"{ascii_word}\n")
            f.write("\n" + "=" * 40 + "\n\n")
