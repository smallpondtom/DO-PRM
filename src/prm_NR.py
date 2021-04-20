import numpy as np
from numba import njit, jit
# from mapping import generate_map, plot_setup


@jit(forceobj=True)
def is_connected_NR(x1, y1, x2, y2, ox, oy, rr):
    """
    Function that connects 2 nodes with no restrictions unless the edge is not 
    intersecting with an obstacle.
    """

    a1 = np.array([x1, y1])
    a2 = np.array([x2, y2])
    b = a2 - a1
    n = b / np.linalg.norm(b)
    for i in range(len(ox)):
        p = [ox[i], oy[i]]
        d = np.linalg.norm(
            (a1 - p) - (np.dot(a1 - p, n))*n
        )
        if d < rr[i]:
            return False
    return True 


@jit(forceobj=True)
def adjacency_mat(x_all, y_all, ox, oy, rr):
    """
    Function that creates the adjacency matrix from the edges and points 
    with the no restriction method 
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
                    if is_connected_NR(x_all[i], y_all[i], x_all[j], y_all[j],
                                       ox, oy, rr):
                        A[i, j] = 1
                        temp.append(j)
        road_map.append(temp)
    A /= A.sum(axis=1, keepdims=True)   # make it row stochastic
    return A, road_map


if __name__=="__main__":
    # start = time.time()
    # Starting point
    sx = 0.0  # x-position
    sy = 0.0  # y-position

    # Goal point
    gx = 10.0  # x-position
    gy = 10.0  # y-position

    # number of obstacles 
    obs_num = 4

    # number of sample points
    N = 20
    # Radial size of the robot
    bot_size = 0.2

    # Obstacles
    ox, oy, rr, x, y = generate_map(sx, sy, gx, gy, obs_num, N, bot_size)

    # Sample points
    plot_setup(sx, sy, gx, gy, ox, oy, rr, x, y)

    x_sample = x 
    y_sample = y
    x.insert(0, sx)
    x.append(gx)
    y.insert(0, sy)
    y.append(gy)

    A = adjacency_mat(x, y, ox, oy, rr)
    edge = list(zip(np.nonzero(A)[0], np.nonzero(A)[1])) 
    print(edge[0])
    # end = time.time()
    # print("Elapsed time: ", (end - start))