import numpy as np
from numpy.linalg import eig

def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """
    #print(x_world.shape)
    #print(x_im.shape)
    # Create the matrix A 
    ##-your-code-starts-here-##
    # Left
    l = np.zeros((2*x_world.shape[0], x_world.shape[1]))
    l[1::2] = x_world

    # Middle
    m = np.zeros((2*x_world.shape[0], x_world.shape[1]))
    m[::2] = x_world

    # Right
    r = np.zeros((2*x_world.shape[0], x_world.shape[1]))
    r[::2] = -x_world*x_im[:,1][:,np.newaxis]
    r[1::2] = -x_world*x_im[:,0][:,np.newaxis]

    A = np.hstack((l,m,r))
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    eigvv = eig(A.T@A)
    i = np.argmin(eigvv[0])
    ev = eigvv[1][i]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    #P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P
