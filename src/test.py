from mapping import generate_map, plot_final, plot_setup, generate_edge, plot_rm
from matrix_utils import is_primitive
from prm_NR import adjacency_mat 
import json 
import time
from dijkstra import dijkstra_planning
import numpy as np
from tqdm import tqdm 


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

# Ideal trajectory 
d_ref = 10*np.sqrt(2)


pbar = tqdm(total=4*5*1000)
case = 1
iterations = 1000
storage = dict()
for N in [10, 20, 50, 100]:
    for obs_num in [2, 4, 6, 8, 10]:
        ct = 0
        invent = dict()
        is_prim_runtime = []
        not_prim_runtime = []
        accuracy = []
        path_found = []
        is_prim = []
        for i in range(iterations):
            start = time.time()
            # Obstacles
            ox, oy, rr, x, y = generate_map(sx, sy, gx, gy, obs_num, N, bot_size)

            # Sample points
            #plot_setup(sx, sy, gx, gy, ox, oy, rr, x, y)

            x_sample = x 
            y_sample = y
            x.insert(0, sx)
            x.append(gx)
            y.insert(0, sy)
            y.append(gy)

            A, road_map = adjacency_mat(x, y, ox, oy, rr)
            rx, ry = dijkstra_planning(sx, sy, gx, gy, road_map, x, y)
            end = time.time()
            dt = end - start 

            if not rx:
                pf = 0
                e = None
            else:
                pf = 1
                # Compute the accuracy 
                d_traj = 0
                for i in range(1, len(rx)):
                    rx1 = rx[i]
                    rx0 = rx[i-1]
                    ry1 = ry[i]
                    ry0 = ry[i-1]
                    dx2 = (rx1 - rx0)**2
                    dy2 = (ry1 - ry0)**2
                    d_traj += np.sqrt(dx2 + dy2)

                e = d_traj - d_ref

            accuracy.append(e)
            path_found.append(pf)
            v = generate_edge(A)

            # Plot final 
            #plot_rm(sx, sy, gx, gy, ox, oy, rr, x, y, v)
            #plot_final(sx, sy, gx, gy, ox, oy, rr, x, y, v, rx, ry)

            if is_primitive(A):
                is_prim_runtime.append(dt)
                is_prim.append(1)
                ct += 1
            else:
                not_prim_runtime.append(dt)
                is_prim.append(0)

            pbar.update(1)
        invent['N'] = N
        invent['Nobs'] = obs_num
        invent['tot-prim'] = ct  
        invent['is_prim_runtime'] = is_prim_runtime
        invent['not_prim_runtime'] = not_prim_runtime
        invent['accuracy'] = accuracy
        invent['path_found'] = path_found
        invent['is_prim'] = is_prim
        storage['case'+str(case)] = invent 
        case += 1

pbar.close()
with open("MC_sim_NR1.json", "+w") as jfile:
    json.dump(storage, jfile, indent=4)

