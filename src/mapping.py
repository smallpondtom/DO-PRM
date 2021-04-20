import random 
import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt 
import scipy
import scipy.spatial
import time 
from typing import Tuple
from prm_NR import adjacency_mat
from matrix_utils import is_primitive


@jit(forceobj=True)
def sample_points(sx, sy, gx, gy, rr, ox, oy, N, bot_size):
    """
    Function that generates sample points based on the starting point
    """
    max_x = gx - 0.1
    max_y = gy - 0.1
    min_x = sx + 0.1
    min_y = sy + 0.1

    sample_x, sample_y = [], []
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
            sample_x.append(tx)
            sample_y.append(ty)
        
    #sample_x.append(sx)
    #sample_y.append(sy)
    #sample_x.append(gx)
    #sample_y.append(gy)

    return sample_x, sample_y


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
                gy: float, n: int, N: int, 
                bot_size: float) -> Tuple[list, list, list, list, list]:
    """
    Function that generates the initial map with obstacles and sample points.
    """

    # Obstacles
    ox, oy, rr = generate_obstacles(n, sx, sy, gx, gy)

    # Sample points
    x, y = sample_points(sx, sy, gx, gy, rr, ox, oy, N, bot_size)
    return ox, oy, rr, x, y


def plot_setup(sx, sy, gx, gy, ox, oy, rr, x, y):
    # Plot the setup
    fig, ax = plt.subplots()
    plt.plot(sx, sy, '.b')
    for oxi, oyi, ori in zip(ox, oy, rr):
        circ = plt.Circle((oxi, oyi), ori, color='k')
        ax.add_patch(circ)
    plt.plot(gx, gy, '*r')
    plt.plot(x, y, '.g')
    ax.set_aspect('equal')
    plt.show()


@jit(forceobj=True)
def generate_edge(A: np.ndarray):
    edge = list(zip(np.nonzero(A)[0], np.nonzero(A)[1])) 
    return edge 


def plot_rm(sx, sy, gx, gy, ox, oy, rr, x, y, edges):
    # Plot the setup
    fig, ax = plt.subplots()
    plt.plot(sx, sy, '.b')
    for oxi, oyi, ori in zip(ox, oy, rr):
        circ = plt.Circle((oxi, oyi), ori, color='k')
        ax.add_patch(circ)
    plt.plot(gx, gy, '*r')
    plt.plot(x, y, '.g')
    for (i, j) in edges:
        if i != j:
            plt.plot([x[i], x[j]], [y[i], y[j]], '-c', lw=0.2)
    ax.set_aspect('equal')
    plt.show()

def plot_final(sx, sy, gx, gy, ox, oy, rr, x, y, edges, rx, ry):
    # Plot the setup
    fig, ax = plt.subplots()
    plt.plot(sx, sy, '.b')
    for oxi, oyi, ori in zip(ox, oy, rr):
        circ = plt.Circle((oxi, oyi), ori, color='k')
        ax.add_patch(circ)
    plt.plot(gx, gy, '*r')
    plt.plot(x, y, '.g')
    for (i, j) in edges:
        if i != j:
            plt.plot([x[i], x[j]], [y[i], y[j]], '-c', lw=0.2)
    plt.plot(rx, ry, '-r', lw=0.25)
    ax.set_aspect('equal')
    plt.show()


if __name__=="__main__":
    # start = time.time()
    # Starting point
    sx = 0.0  # x-position
    sy = 0.0  # y-position

    # Goal point
    gx = 10.0  # x-position
    gy = 10.0  # y-position

    # # number of obstacles 
    # obs_num = 1

    # # number of sample points
    # n = 10
    # Radial size of the robot
    bot_size = 0.2

    case = 1
    iterations = 1000
    storage = dict()
    for N in [20]:
        for obs_num in [2, 4, 6, 8, 10]:
            ct = 0
            invent = dict()
            for i in range(iterations):
                # Obstacles
                ox, oy, rr, x, y = generate_map(sx, sy, gx, gy, obs_num, N,
                                                bot_size)

                # Sample points
                #plot_setup(sx, sy, gx, gy, ox, oy, rr, x, y)

                x_sample = x 
                y_sample = y
                x.insert(0, sx)
                x.append(gx)
                y.insert(0, sy)
                y.append(gy)

                A = adjacency_mat(x, y, ox, oy, rr)
                # v = generate_edge(A)


                # plot_rm(sx, sy, gx, gy, ox, oy, x, y, v)
                if is_primitive(A):
                    ct += 1
            invent['N'] = N
            invent['Nobs'] = obs_num
            invent['tot-prim'] = ct  
            storage[str(case)] = invent 
            case += 1


    # plot_rm(sx, sy, gx, gy, ox, oy, x, y, v)

    # end = time.time()
    # print("Elapsed time: ", (end - start))
