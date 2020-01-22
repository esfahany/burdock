# sweep operator

# sweeps a covariance matrix to extract regression coefficients

import numpy as np

def amsweep(g, m):
    """
    Sweeps a covariance matrix to extract regression coefficients.

    Args:
        g (Numpy array): a numeric, symmetric covariance matrix divided by the number of observations in the data
        m (Numpy array): a logical vector of length equal to the number of rows in g 
        in which the True values correspond to the x values in the matrix 
        and the False values correspond to the y values in the matrix

    Return:
        a matrix with the coefficients from g
    """

    # if m is a vector of all falses, then return g
    if np.array_equal(m, np.full(np.shape(m), False, dtype=bool)): 
        return g
    
    else:
        p = np.shape(g)[0] # number of rows of g (np.shape gives a tuple as (rows, cols), so we index [0])
        rowsm = sum(m) # sum of logical vector "m" (m must be a (n,) shape np array)

        if p == rowsm: # if all values of m are True (thus making the sum equal to the length), we take the inverse and then negate all the values
            h = np.linalg.inv(g) # inverse of g
            h = np.negative(h) # negate the sign of all elements

        else:
            k = np.where(m == True)[0] # indices where m is True
            kcompl = np.where(m == False)[0] # indices where m is False

            # separate the elements of g
            # make the type np.matrix so that dimensions are preserved correctly
            g11 = np.matrix(g[k, k])
            g12 = np.matrix(g[k, kcompl])
            g21 = np.transpose(g12)
            g22 = np.matrix(g[kcompl, kcompl])
            
            # use a try-except to get the inverse of g11
            try:
                h11a = np.linalg.inv(g11) # try to get the regular inverse
            except: # should have LinAlgError (not defined error)
                h11a = np.linalg.pinv(g11)
            h11 = np.negative(h11a)

            # matrix multiplication to get sections of h
            h12 = np.matmul(h11a, g12)
            h21 = np.transpose(h12)
            h22 = g22 - np.matmul(np.matmul(g21, h11a), g12)

            # combine sections of h
            hwo = np.concatenate((np.concatenate((h11, h12), axis = 1), np.concatenate((h21, h22), axis = 1)), axis = 0)
            hwo = np.asarray(hwo) # convert back to array (from matrix) to avoid weird indexing behavior

            xordering = np.concatenate((k, kcompl), axis = 0) # concatenate k and kcompl

            h = np.zeros((p, p)) # make a pxp array of zeros
            for i in range(p): # traverse each element as defined by xordering
                for j in range(p):
                    h[xordering[i]][xordering[j]] = hwo[i][j] # and replace it with the normal i, j element from hwo

        return h
