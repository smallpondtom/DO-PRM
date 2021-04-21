import numpy as np
from numba import njit, jit
import random 
import scipy.spatial
from typing import Tuple
# from mapping import generate_map, plot_setup


@jit(forceobj=True)
def get_radius(rr, field_len, N):
    mu_forb = 0
    for r in rr:
        mu_forb += np.pi * r**2
    mu_free = field_len**2 - mu_forb 
    zeta_b = np.pi
    d = 2  # 2-dimension 
    gamma_prm_star = 2 * (1 + 1/d)**(1/d) * (mu_free/zeta_b)**(1/d)
    gamma_prm = (1 + random.random()) * gamma_prm_star
    if N == 0:
        return 1
    else:
        return gamma_prm * (np.log(N)/N)**(1/d)


@jit(forceobj=True)
def is_connected_star(x1, y1, x2, y2, ox, oy, rr, R1, R2):
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
        # if the 2 points are not within a ball of the varying radius
        # for the 2 points
        if (R1 < b_mag) and (R2 < b_mag):  
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
                    if is_connected_star(x_all[i], y_all[i], x_all[j], y_all[j],
                                         ox, oy, rr, R[i], R[j]):
                        A[i, j] = 1
                        temp.append(j)
        road_map.append(temp)
    A /= A.sum(axis=1, keepdims=True)   # make it row stochastic
    return A, road_map


@jit(forceobj=True)
def sample_points(sx, sy, gx, gy, rr, ox, oy, N, bot_size, fl):
    """
    Function that generates sample points based on the starting point
    """
    max_x = gx - 0.1
    max_y = gy - 0.1
    min_x = sx + 0.1
    min_y = sy + 0.1

    sample_x, sample_y = [], []
    R = []  # varying radius 

    obstacle_kd_tree = scipy.spatial.cKDTree(np.vstack((ox, oy)).T)
    while len(sample_x) <= N-1:
        tx = round((random.random() * (max_x - min_x)) + min_x, 4) 
        ty = round((random.random() * (max_y - min_y)) + min_y, 4) 

        dist, index = obstacle_kd_tree.query([tx, ty], k=len(ox))
        ct = 0
        for i in range(len(dist)):
            if dist[i] - rr[index[i]] >= bot_size:
                ct += 1
        if ct == len(dist):
            Ri = get_radius(rr, fl, len(sample_x))
            R.append(Ri)
            sample_x.append(tx)
            sample_y.append(ty)
        
    #sample_x.append(sx)
    #sample_y.append(sy)
    #sample_x.append(gx)
    #sample_y.append(gy)

    return sample_x, sample_y, R


@jit(forceobj=True)
def generate_obstacles(n: int, sx: float, sy: float,
                        gx: float, gy: float) -> Tuple[list, list, list]:
    """
    Function that generates obstacles in the field.
    :param n = <int> number of nodes 
    :param sx = <float> x-position of starting point
    :param sy = <float> y-position of starting point
    :param gx = <float> x-position of goal point
    :param gy = <float> y-position of goal point
    :output1 = <list> x-positions of obstacles 
    :output2 = <list> y-positions of obstacles  
    :output3 = <list> radius of each circular obstacle
    """ 
    
    temp = np.array([[sx, sy], [gx, gy]])
    while True: 
        ox = []
        oy = []
        rr = []
        
        for i in range(n):
            ox.append(round(random.random() * (gx - sx) + sx, 4))
            oy.append(round(random.random() * (gy - sy) + sy, 4))
            rr.append(round(random.random() * (1.0 - 0.2) + 0.2, 4))
        

        dist1 = scipy.spatial.distance.cdist(
                np.vstack((ox, oy)).T, temp
                )
        dist2 = scipy.spatial.distance.cdist(
                np.vstack((ox, oy)).T, np.vstack((ox, oy)).T
                )

        for i in range(len(dist2)):
            for j in range(len(dist2[0])):
                if i != j:
                    dist2[i, j] -= rr[i] + rr[j]
        if np.all(dist1 >= rr[i]) and np.all(dist2 >= 0):
            break

    return ox, oy, rr


@jit(forceobj=True)
def generate_map(sx: float, sy: float, gx: float, 
                gy: float, fl: int, n: int, N: int, 
                bot_size: float) -> Tuple[list, list, list, list, list, list]:
    """
    Function that generates the initial map with obstacles and sample points.
    """

    # Obstacles
    ox, oy, rr = generate_obstacles(n, sx, sy, gx, gy)

    # Sample points
    x, y, R = sample_points(sx, sy, gx, gy, rr, ox, oy, N, bot_size, fl)
    return ox, oy, rr, x, y, R
