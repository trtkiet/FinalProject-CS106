import numpy as np

def gram_matrix_linear(X):
    return np.dot(X, X.T)

def gram_matrix_rbf(X, sigma=1.0):
    """
    Compute the RBF (Gaussian) kernel Gram matrix.
    
    Parameters:
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    sigma : float
        Bandwidth parameter for the RBF kernel.
        
    Returns:
    np.ndarray
        The RBF kernel Gram matrix of shape (n_samples, n_samples).
    """
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-sq_dists / (2 * sigma**2))

def calc_partion(A, x, y, u, v):
    sum = A[u, v]
    if x > 0:
        sum -= A[x-1, v]
    if y > 0:
        sum -= A[u, y-1]
    if x > 0 and y > 0:
        sum += A[x-1, y-1]
    return sum
    
def penalty(m, n):
    if m == 0:
        return 0
    return m * (np.log(n / m) + 1)

def Kernel_temporal_segmentation(features, max_change_points=20, penalty_factor=0.01):
    n = features.shape[0]
    
    A = gram_matrix_linear(features)
    
    A_cum = np.cumsum(np.cumsum(A, axis=0), axis=1)
    v = np.zeros((n, n+1), dtype=np.float32)
    for t in range(n):
        sum_diag = A[t, t]
        for d in range(1, n - t + 1):
            sum_diag += A[t + d - 1, t + d - 1]
            
            sum_block = calc_partion(A_cum, t, t, t + d - 1, t + d - 1)
            v[t, t + d] = sum_block - (1.0 / d) * sum_diag
    
    L = np.full((max_change_points+1, n+1), np.inf, dtype=np.float32)
    for i in range(max_change_points + 1):
        if i == 0:
            L[i, :] = v[0, :]
            continue
        for j in range(1, n + 1):
            for t in range(i, j):
                L[i, j] = min(L[i, j], L[i - 1, t] + v[t, j])
    
    m = 0
    for i in range(1, max_change_points + 1):
        if L[m, n] + penalty_factor * penalty(m, n) > L[i, n] + penalty_factor * penalty(i, n):
            m = i

    change_points = []
    current_end = n
    current_m = m
    
    while current_m > 0 and current_end > 0:
        best_t = current_m
        
        for t in range(current_m, current_end):
            if L[current_m-1, t] != np.inf and v[t, current_end] != np.inf:
                cost = L[current_m-1, t] + v[t, current_end]
                if abs(cost - L[current_m, current_end]) < 1e-10:  # Found the optimal split
                    best_t = t
                    break
        
        change_points.append(best_t)
        current_end = best_t
        current_m -= 1
    
    # Reverse to get chronological order and convert to 0-based indexing
    change_points.reverse()
    return np.array(change_points, dtype=int)