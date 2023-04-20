import numpy as np
def t2b(A, B, C):
    norma = 0
    for a in A:
        norma += a**2
    norma = norma**0.5
    
    normb = 0
    for b in B:
        normb += b**2
    normb = normb**0.5
    
    normc = 0
    for c in C:
        normc += c**2
    normc = normc**0.5
        
    dotab = 0
    for a,b in zip(A,B):
        dotab += a*b
        
    dotac = 0
    for a,c in zip(A,C):
        dotac += a*c

    
    cosab = dotab / (norma*normb)
    cosac = dotac / (norma*normc)
    
    if 1-cosab < 1-cosac:
        return B
    
    return C

#%% part 4 22
from numpy.linalg import det
def cornerresponse(M):
    a = 0.05
    return det(M) - a*np.trace(M)**2
