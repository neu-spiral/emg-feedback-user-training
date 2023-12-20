from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

from emg_feedback_user_training.utils import NUM_CLASSES


def normalize_sims(all_sims, normalized):
    min_overall = np.nanmin(all_sims)
    max_overall = np.nanmax(all_sims)

    if normalized:
        all_sims = [d - min_overall for d in all_sims]
        all_sims = [d / (max_overall - min_overall) for d in all_sims]
    return all_sims


def scalar_similarity_measure(pairwise_similarities: np.ndarray, within_class_numerator: bool):
    avg_within_class = np.nanmean(np.diag(pairwise_similarities))
    # Symmetric matrix, so we only need lower tri
    tril = pairwise_similarities[np.tril_indices(pairwise_similarities.shape[0], -1)]
    avg_across_class = np.nanmean(tril)
    if within_class_numerator:
        return avg_within_class / avg_across_class
    else:
        return avg_across_class / avg_within_class


def median_heuristic_gamma(data):
    """
    See https://github.com/eugenium/MMD/blob/master/mmd.py, https://arxiv.org/pdf/1707.07269.pdf:

    "Pick sigma in the expression rbf(x, y) = exp(- ||x-y||^2 / (2*sigma^2)),
    so that ||x-y||^2 / (2*sigma2)  equals 1 for the median similarity"

    RBF kernel:

        rbf(x_i, x_j) = exp( - gamma || x_i - x_j ||^2 )
                      = exp( - 1/(2 sigma^2) || x_i - x_j ||^2 )

    The rule of thumb:

        sigma = sqrt(median / 2)
        gamma = 1 / (2 sigma^2)
        gamma = 1 / (2 sqrt(median / 2) ^2)
        gamma = 1 / median
    """
    similarities = pdist(data, "sqeuclidean")
    return 1 / np.median(similarities)


def compute_feature_similarities(
    features, one_hot_labels, method: str, gamma: float, n_class=NUM_CLASSES
) -> np.ndarray:
    """Computes some notion of similarity between each pair of classes in feature space."""

    def d_bhat(m1, m2, c1, c2):
        # NOTE - probably not numerically stable way to compute this
        # especially since cov matrices are probably not well conditioned
        sigma = c1 + c2 / 2
        term1 = 0.125 * (m1 - m2).T @ np.linalg.inv(sigma) @ (m1 - m2)
        term2 = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(np.linalg.det(c1) * np.linalg.det(c2)))
        return term1 + term2

    labels = one_hot_labels.argmax(-1)
    # NOTE - start with all NaN, so that missing gestures will not skew the average towards zero
    # or some other default value. Later, need to use functions like np.nanmean(), etc
    pairwise_similarities = np.nan * np.ones((n_class, n_class))

    if method == "avg_pairwise_l2":
        # Avg between_class similarities
        for label1, label2 in combinations(np.unique(labels), 2):
            subset1 = features[labels == label1]
            subset2 = features[labels == label2]
            pairwise_sims = cdist(subset1, subset2, "euclidean")
            pairwise_similarities[label1, label2] = pairwise_similarities[label2, label1] = pairwise_sims.mean()
        # Fill diagonal.  different because we must ignore zeros due to d(x1, x1) == 0
        for label in np.unique(labels):
            subset = features[labels == label]
            if len(subset) <= 1:
                continue
            pairwise_similarities[label, label] = pdist(subset, "euclidean").mean()

    elif method == "rbf":
        # RBF Kernel similarities
        # First, find the median of all pairwise L2 similarities
        for label1, label2 in combinations(np.unique(labels), 2):
            subset1 = features[labels == label1]
            subset2 = features[labels == label2]
            rbf_similarities = rbf_kernel(subset1, subset2, gamma=gamma)
            pairwise_similarities[label1, label2] = pairwise_similarities[label2, label1] = rbf_similarities.mean()
        # Fill diagonal.  different because we must ignore zeros due to d(x1, x1) == 0
        for label in np.unique(labels):
            subset = features[labels == label]
            if len(subset) <= 1:
                continue
            rbf_similarities = np.exp(-gamma * pdist(subset, "sqeuclidean"))
            pairwise_similarities[label, label] = rbf_similarities.mean()

    elif method == "cos":
        for label1, label2 in combinations(np.unique(labels), 2):
            subset1 = features[labels == label1]
            subset2 = features[labels == label2]
            cos_similarities = cosine_similarity(subset1, subset2)
            pairwise_similarities[label1, label2] = pairwise_similarities[label2, label1] = cos_similarities.mean()
        # Fill diagonal.  different because we must ignore zeros due to d(x1, x1) == 0
        for label in np.unique(labels):
            subset = features[labels == label]
            if len(subset) <= 1:
                continue
            cos_similarities = cosine_similarity(subset)
            pairwise_similarities[label, label] = cos_similarities.mean()

    elif method == "ncut":
        # Similarity notion, based on normalized cuts
        # https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
        rbf = rbf_kernel(features, gamma=gamma)
        # Compute normalized cut between each pair of classes.
        # For classes A and B, the normalized cut is defined as:
        # n_cut(A, B) = cut(A, B) / assoc(A, V) + cut(A, B) / assoc(B, V)
        # where cut(A, B) is the total weight of edges between A and B
        # and assoc(A, V) is the total weight of edges between A and all vertices outside of A
        for label1, label2 in combinations(np.unique(labels), 2):
            idx1 = labels == label1
            idx2 = labels == label2
            # Compute total volume of both classes
            assoc1 = rbf[idx1, :].sum()
            assoc2 = rbf[idx2, :].sum()
            # Compute cut between classes
            cut = rbf[np.ix_(idx1, idx2)].sum() + rbf[np.ix_(idx2, idx1)].sum()
            # Compute normalized cut
            ncut = (cut / assoc1) + (cut / assoc2)
            pairwise_similarities[label1, label2] = pairwise_similarities[label2, label1] = ncut
        for label in np.unique(labels):
            idx = labels == label
            if len(idx) <= 1:
                continue
            assoc_within = rbf[np.ix_(idx, idx)].sum()
            assoc_total = rbf[idx, :].sum()
            assoc_outside = assoc_total - assoc_within
            pairwise_similarities[label, label] = assoc_within / assoc_outside

    elif method == "mahalanobis":
        # Note that Mahalanobis similarity is computed between a pair of points
        means, covs = [], []
        regularization = 0.01
        all_cov = np.cov(features.T)
        inv_cov = np.linalg.inv(all_cov + regularization * np.trace(all_cov) * np.eye(all_cov.shape[0]))

        for label1, label2 in combinations(np.unique(labels), 2):
            subset1 = features[labels == label1]
            subset2 = features[labels == label2]
            d = cdist(subset1, subset2, metric="mahalanobis", VI=inv_cov)
            pairwise_similarities[label1, label2] = pairwise_similarities[label2, label1] = d.mean()
        for label in np.unique(labels):
            subset = features[labels == label]
            if len(subset) <= 1:
                continue
            d = pdist(subset, metric="mahalanobis", VI=inv_cov)
            pairwise_similarities[label, label] = d.mean()

    elif method == "bhattacharyya":
        # Note that this similarity is computed for each pair of estimated distrbutions
        means, covs = [], []
        regularization = 0.01
        for label in np.unique(labels):
            subset = features[labels == label]
            means.append(subset.mean(axis=0))
            c = np.cov(subset.T)
            covs.append(c + regularization * np.trace(c) * np.eye(c.shape[0]))

        for label1, label2 in combinations(np.unique(labels), 2):
            d = d_bhat(means[label1], means[label2], covs[label1], covs[label2])
            pairwise_similarities[label1, label2] = pairwise_similarities[label2, label1] = d

    elif method == "kde_kl":
        raise NotImplementedError()
        # for label1, label2 in combinations(np.unique(labels), 2):
        # Estimate KL divergence between the two groups using KDE fit to each one.
        # subset1 = features[labels == label1]
        # subset2 = features[labels == label2]
        # k1 = KernelDensity().fit(subset1)
        # k2 = KernelDensity().fit(subset2)

    else:
        raise ValueError(f"Unknown method: {method}")

    return pairwise_similarities
