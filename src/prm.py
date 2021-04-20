import numpy as np
from numba import njit, jit
# from mapping import generate_map, plot_setup



@jit(forceobj=True)
def is_connected(x1, y1, x2, y2, ox, oy, rr, R):
    """
    Function that connects 2 nodes with a fixed radius unless the edge is not 
    intersecting with an obstacle.
    """

    a1 = np.array([x1, y1])
    a2 = np.array([x2, y2])
    b = a2 - a1
    b_mag = np.linalg.norm(b)
    n = b / b_mag
    for i in range(len(ox)):
        p = [ox[i], oy[i]]
        d = np.linalg.norm(
            (a1 - p) - (np.dot(a1 - p, n))*n
        )
        if d < rr[i]:
            return False

        if R < b_mag:  # if the 2 points are not within a ball of fixed radius 
            return False
    return True 


@jit(forceobj=True)
def adjacency_mat(x_all, y_all, ox, oy, rr, R):
    """
    Function that creates the adjacency matrix from the edges and points 
    with the fixed Radius method or conventional PRM method 
    """
    n = len(x_all)
    A = np.zeros((n, n))
    road_map = []
    for i in range(n):
        temp = []
        for j in range(n):
            if i == j:
                A[i, j] = 1
            else:
                if A[j, i] != 0:
                    A[i, j] = A[j, i]
                    temp.append(j) 
                else:
                    if is_connected(x_all[i], y_all[i], x_all[j], y_all[j],
                                       ox, oy, rr, R):
                        A[i, j] = 1
                        temp.append(j)
        road_map.append(temp)
    A /= A.sum(axis=1, keepdims=True)   # make it row stochastic
    return A, road_map


