import numpy as np
def local_pca(X,k=3):
    u,s,v=np.linalg.svd(X,full_matrices=False); return s[:k]
