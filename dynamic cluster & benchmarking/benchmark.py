# benchmark.py

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture

from em_algorithm import run_em  # Your custom EM

df = pd.read_csv("/Users/anirudhjp/Downloads/people_wiki.csv")  # ✅ Use your full path here
documents = df["text"].tolist()

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(documents)
X = normalize(X, norm='l2')

print("\nRunning custom EM...")
start = time.time()
responsibilities, weights, means, variances, log_likelihoods = run_em(X, K=5, max_iter=20)
end = time.time()

em_total_time = end - start
em_iter_count = len(log_likelihoods)
em_time_per_iter = em_total_time / em_iter_count
em_final_ll = log_likelihoods[-1]

#Benchmark: Sklearn GMM

print("\nRunning sklearn GMM...")
X_dense = X.toarray()  # sklearn requires dense input
start = time.time()
gmm = GaussianMixture(n_components=5, covariance_type='diag', max_iter=20, n_init=5)

gmm.fit(X_dense)
end = time.time()

sklearn_total_time = end - start
sklearn_iter_count = gmm.n_iter_
sklearn_time_per_iter = sklearn_total_time / sklearn_iter_count
sklearn_log_likelihood = gmm.lower_bound_ * X_dense.shape[0]

#Print Comparison Table

print("\n=== Benchmark Comparison ===")
print(f"{'Metric':<25} | {'Your EM':<15} | {'Sklearn GMM':<15}")
print("-" * 60)
print(f"{'Total Time (s)':<25} | {em_total_time:<15.2f} | {sklearn_total_time:<15.2f}")
print(f"{'Iterations':<25} | {em_iter_count:<15} | {sklearn_iter_count:<15}")
print(f"{'Time per Iter (s)':<25} | {em_time_per_iter:<15.2f} | {sklearn_time_per_iter:<15.2f}")
print(f"{'Final Log-Likelihood':<25} | {em_final_ll:<15.2f} | {sklearn_log_likelihood:<15.2f}")

with open("benchmark_results.txt", "w") as f:
    f.write("=== Benchmark Comparison ===\n")
    f.write(f"{'Metric':<25} | {'Your EM':<15} | {'Sklearn GMM':<15}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Total Time (s)':<25} | {em_total_time:<15.2f} | {sklearn_total_time:<15.2f}\n")
    f.write(f"{'Iterations':<25} | {em_iter_count:<15} | {sklearn_iter_count:<15}\n")
    f.write(f"{'Time per Iter (s)':<25} | {em_time_per_iter:<15.2f} | {sklearn_time_per_iter:<15.2f}\n")
    f.write(f"{'Final Log-Likelihood':<25} | {em_final_ll:<15.2f} | {sklearn_log_likelihood:<15.2f}\n")
