# em_visualizer.py

import curses
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from em_algorithm import compute_log_likelihood_sparse, m_step  # Make sure this import works

def run_em_with_visualization(tfidf_matrix, K=5, max_iter=20, threshold=1e-4):
    def draw_screen(stdscr):
        curses.curs_set(0)
        stdscr.clear()

        # KMeans Initialization
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        assignments = kmeans.fit_predict(tfidf_matrix)
        means = kmeans.cluster_centers_
        N, D = tfidf_matrix.shape
        weights = np.array([(assignments == k).sum() for k in range(K)]) / N

        # Variance Initialization
        variances = np.zeros((K, D))
        for k in range(K):
            cluster_points = tfidf_matrix[assignments == k]
            if cluster_points.shape[0] > 0:
                mean_vec = sp.csr_matrix(means[k])
                mean_matrix = sp.vstack([mean_vec] * cluster_points.shape[0])
                diff = cluster_points - mean_matrix
                var = diff.multiply(diff).mean(axis=0)
                variances[k] = np.asarray(var).flatten()
            variances[k][variances[k] < 1e-8] = 1e-8

        log_likelihoods = []

        for iteration in range(max_iter):
            log_probs = compute_log_likelihood_sparse(tfidf_matrix, means, variances, weights)
            log_sum = np.logaddexp.reduce(log_probs, axis=1)
            responsibilities = np.exp(log_probs - log_sum[:, np.newaxis])
            log_likelihood = np.sum(log_sum)
            log_likelihoods.append(log_likelihood)

            # --- Draw to terminal screen ---
            stdscr.clear()
            stdscr.addstr(0, 0, f"EM Iteration {iteration + 1}")
            stdscr.addstr(1, 0, f"Log-likelihood: {log_likelihood:.4f}")
            
            if iteration > 0:
                delta = log_likelihoods[-1] - log_likelihoods[-2]
                stdscr.addstr(2, 0, f"Δ Log-likelihood: {delta:.4f}")

                # Moving average over last 5 deltas
                recent = log_likelihoods[-5:] if len(log_likelihoods) >= 5 else log_likelihoods
                avg_delta = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) > 1 else delta
                stdscr.addstr(3, 0, f"Moving Avg (last {len(recent)}): {avg_delta:.4f}")

            for k in range(K):
                stdscr.addstr(5 + k, 0, f"Cluster {k}: weight = {weights[k]:.4f}")

            stdscr.refresh()
            time.sleep(0.5)

            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < threshold:
                break

            weights, means, variances = m_step(tfidf_matrix, responsibilities)

    curses.wrapper(draw_screen)

# Entry point
if __name__ == "__main__":
    print("Loading data and preparing TF-IDF matrix...")

    df = pd.read_csv("/Users/anirudhjp/Downloads/people_wiki.csv")  # <- YOUR FULL PATH
    documents = df["text"].tolist()

    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_matrix = normalize(tfidf_matrix, norm='l2')

    print("Starting EM with terminal visualization...")
    run_em_with_visualization(tfidf_matrix)
