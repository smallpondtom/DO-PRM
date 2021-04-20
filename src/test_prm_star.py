from matrix_utils import is_primitive
from prm_star import generate_map, adjacency_mat
import json 


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

for N in [10, 20, 50, 100]:
    for obs_num in [2, 4, 6, 8, 10]:
        ct = 0
        invent = dict()
        for i in range(iterations):
            # Obstacles
            ox, oy, rr, x, y, R = generate_map(sx, sy, gx, gy, 10, obs_num, N, bot_size)

            # Sample points
            # plot_setup(sx, sy, gx, gy, ox, oy, rr, x, y)

            x_sample = x 
            y_sample = y
            x.insert(0, sx)
            x.append(gx)
            y.insert(0, sy)
            y.append(gy)

            A = adjacency_mat(x, y, ox, oy, rr, R)
            # v = generate_edge(A)
            if is_primitive(A):
                ct += 1
        invent['N'] = N
        invent['Nobs'] = obs_num
        invent['tot-prim'] = ct  
        storage[str(case)] = invent 
        case += 1

with open("MC_sim_PRM_star.json", "w") as jfile:
    json.dump(storage, jfile, indent=4)

