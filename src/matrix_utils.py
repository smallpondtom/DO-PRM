import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt



def getmat():
    """
    Function that creates a matrix from the user inputs.
    """

    # Get the number of ns and columns from user
    n = int(input("Enter the dimensions of your square matrix: "))

    # Initiate the matrix as empty list
    mat = []
    for i in range(n):
        flag = True
        print("Enter the non-negative elements for n #{} as a single line separated by spaces: ".format(i+1))
        while flag:  # add some input validation for robustness
            temp_row = list(map(float, input().split()))
            temp_np_row = np.array(temp_row)
            sz = temp_np_row.shape[0]
            # condition of having n columns and all positive values
            if sz == n and not np.any(temp_np_row < 0):
                flag = False
                mat.append(temp_row)
            elif np.any(temp_np_row < 0):  # condition where there is a negative value
                print("Error. Enter only positive values. Try again.")
            else:  # condition where the number of columns are not appropriate
                print(
                    "Error. Enter only {} elements corresponding to your number of columns. Try again.")
    return np.array(mat)


def gershgorin(A):
    """
    Function that computes all the Gershgorin disks with center point and radius.
    """

    # Get the diagonal entries of the A matrix which are the centers of the Gershgorin disk
    centers = np.diagonal(A)

    # Initialize array that stores the radii for corresponding disks
    radii = np.zeros(A.shape[0])

    # Loop that calculates the radius for each Gershgorin disk
    for idx, row_vals in enumerate(A):
        temp_row = np.delete(row_vals, idx)
        radii[idx] = np.sum(temp_row)
    return centers, radii


def plot_disks(centers, radii):
    """
    Function that plots the Gershgorin disks
    """
    fig, axes = plt.subplots()
    # Upper and lower limit for the figure axis
    upperlim = centers[0]+radii[0]
    for i in range(centers.shape[0]):
        C = plt.Circle((centers[i], 0), radii[i], fill=False, color='b')
        axes.add_artist(C)
        if centers[i]+radii[i] > upperlim:  # determine the upper bound for the axis size
            upperlim = centers[i] + radii[i]

    axes.axhline(y=0, color='k')
    axes.axvline(x=0, color='k')
    plt.title('Gershgorin Disks for Matrix A')
    plt.xlim((-upperlim, upperlim))
    plt.ylim((-upperlim, upperlim))
    axes.minorticks_on()  # turn on minor ticks which is required 
    axes.grid(which='major', color='k', linestyle='--')
    axes.grid(which='minor', color='k', linestyle=':')
    axes.set_aspect(1)
    plt.show()


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


@jit(forceobj=True)
def is_primitive(A):
    n = len(A)
    if np.all(np.linalg.matrix_power(A, n**2-2*n+2) > 0):
        return True
    else:
        return False


if __name__ == "__main__":
    A = getmat()
    c, r = gershgorin(A)
    plot_disks(c, r)
