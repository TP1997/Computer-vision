import numpy as np

def proj(X, P):
    x = P[0,0]*X[0]/X[2]
    y = P[0,0]*X[1]/X[2]
    
    return (x,y)

#%%
x1 = np.array([2,4,1])
x2 = np.array([8,8,1])
l1 = np.cross(x1, x2)
#%%
x1 = np.array([14,10,1])
x2 = np.array([18,6,1])
l2 = np.cross(x1, x2)