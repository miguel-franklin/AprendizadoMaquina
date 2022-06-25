import numpy as np


def pca(X, dimension=0, normalise=True):
    mean = np.mean(X, axis=0)
    X -= mean

    cov = np.cov(np.transpose(X))

    evals, evecs = np.linalg.eig(cov)
    idx = np.argsort(evals)
    idx = idx[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    if dimension > 0:
        evecs = evecs[:, :dimension]

    if normalise:
        for i in range(np.shape(evecs)[1]):
            evecs[:, i] / np.linalg.norm(evecs[:, i]) * np.sqrt(evals[i])

    x = np.dot(np.transpose(evecs), np.transpose(X))

    y = np.transpose(np.dot(evecs, x)) + mean
    return x, y, evals, evecs
