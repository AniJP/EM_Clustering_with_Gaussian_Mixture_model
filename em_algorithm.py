import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
from sklearn.cluster import KMeans

def compute_log_likelihood_sparse(tfidf_matrix, means, variances, weights):
    N, D = tfidf_matrix.shape
    K = means.shape[0]
    log_probs = np.zeros((N, K))

    for k in range(K):
        log_det = np.sum(np.log(variances[k]))
        const = -0.5 * (np.log(2 * np.pi) * D + log_det)

        inv_var = 1.0 / variances[k]
        x_sq_scaled = tfidf_matrix.multiply(tfidf_matrix).dot(inv_var)
        cross_term = tfidf_matrix.dot(means[k] * inv_var)
        mu_sq_term = np.sum((means[k] ** 2) * inv_var)

        mahalanobis = x_sq_scaled - 2 * cross_term + mu_sq_term
        log_probs[:, k] = const - 0.5 * mahalanobis + np.log(weights[k])

    return log_probs

def m_step(tfidf_matrix, responsibilities):
    N, D = tfidf_matrix.shape
    K = responsibilities.shape[1]

    Nk = responsibilities.sum(axis=0)
    weights = Nk / N

    means = np.zeros((K, D))
    for k in range(K):
        resp = responsibilities[:, k]
        weighted_sum = tfidf_matrix.T.dot(resp)
        means[k] = (weighted_sum / Nk[k]).A1 if sp.issparse(weighted_sum) else weighted_sum / Nk[k]

    variances = np.zeros((K, D))
    for k in range(K):
        mu = means[k]
        resp = responsibilities[:, k]

        x_sq = tfidf_matrix.multiply(tfidf_matrix)
        x_sq_weighted_sum = x_sq.T.dot(resp)
        ex2 = x_sq_weighted_sum / Nk[k]
        var = ex2 - mu ** 2
        var[var < 1e-8] = 1e-8
        variances[k] = var

    return weights, means, variances

def run_em(tfidf_matrix, K=5, max_iter=20, threshold=1e-4):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(tfidf_matrix)
    means = kmeans.cluster_centers_
    N, D = tfidf_matrix.shape
    weights = np.array([(assignments == k).sum() for k in range(K)]) / N

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
        log_sum = logsumexp(log_probs, axis=1)
        log_responsibilities = log_probs - log_sum[:, np.newaxis]
        responsibilities = np.exp(log_responsibilities)
        log_likelihood = np.sum(log_sum)
        log_likelihoods.append(log_likelihood)

        print(f"EM Iteration {iteration + 1}: Log-likelihood = {log_likelihood:.4f}")
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < threshold:
            print("Converged.")
            break

        weights, means, variances = m_step(tfidf_matrix, responsibilities)

    return responsibilities, weights, means, variances, log_likelihoods
