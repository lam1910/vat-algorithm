import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

def vat(input_matrix):
    #distance_matrix = np.matmul(distance_matrix, distance_matrix.T)
    distance_matrix = ssd.squareform(ssd.pdist(input_matrix, 'euclidean'))
    n = distance_matrix.shape[0]
    J = np.arange(n)
    I = [0]
    J = np.delete(J, 0)
    P = [0]
    for r in range(1, n):
        i = I[-1]
        d = distance_matrix[i, J]
        j = J[np.argmin(d)]
        I.append(j)
        J = np.delete(J, np.where(J == j))
        P.append(j)
    return distance_matrix[np.ix_(P, P)]
    
def save_map(map_data, fig_name='vat-reorder.png', fig_type='png'):
    fig, ax = plt.subplots()
    ax.imshow(map_data, cmap='gray')
    ax.set_title('VAT Reordered Distance Matrix')
    fig.savefig(fig_name, format=fig_type)