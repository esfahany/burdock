# sweep operator

# sweeps a covariance matrix to extract regression coefficients
import numpy as np

def amsweep(g, k, m):
    """
    g is a numeric, symmetric covariance matrix divided by the number of observations in the data 

    k is col

    m is a logical vector of length equal to the number of rows 
    """

    if m == np.full(np.shape(m), False, dtype=bool): 
        return 
        
    g = np.asarray(g)

    h = g - np.outer(g[:, k], g[k, :]) / g[k, k]
    # h = g - g[:, k:k+1] * g[k, :] / g[k, k]
    # Modify the k-th row and column
    h[:, k] = g[:, k] / g[k, k]
    h[k, :] = h[:, k]
    # Modify the pivot
    h[k, k] = -1 / g[k, k]

    return h
    