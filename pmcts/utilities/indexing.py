import numpy as np


def xy_to_ij(xy, extent, max_row, max_col):
    # x -> j
    j = (xy[:, 0] - extent[0]) / (extent[1] - extent[0]) * max_col
    j = np.round(j, decimals=6)
    j[j < 0] = 0
    j[j > max_col] = max_col
    # y -> i
    i = (xy[:, 1] - extent[2]) / (extent[3] - extent[2]) * max_row
    i = np.round(i, decimals=6)
    i[i < 0] = 0
    i[i > max_row] = max_row
    # stack
    ij = np.vstack([i.ravel(), j.ravel()]).T.astype(np.int)
    return ij
