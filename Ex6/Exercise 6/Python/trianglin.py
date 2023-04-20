import numpy as np
from numpy.linalg import eig

def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    x1cpm = np.array([0,-x1[2],x1[1],x1[2],0,-x1[0],-x1[1],x1[0],0]).reshape((3,3))
    x1cpmP1 = x1cpm@P1
    x2cpm = np.array([0,-x2[2],x2[1],x2[2],0,-x2[0],-x2[1],x2[0],0]).reshape((3,3))
    x2cpmP2 = x2cpm@P2
    
    A = np.vstack((x1cpmP1,x2cpmP2))
    
    eigvv = eig(A.T@A)
    i = np.argmin(eigvv[0])
    ev = eigvv[1][i]
    ##-your-code-ends-here-##

    X = ev #np.array([0, 0, 0, 1])  # remove me
    
    return X
