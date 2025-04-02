import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

def vat(input_matrix):
    # Get the distance matrix from the input matrix (step 1)
    distance_matrix = ssd.squareform(ssd.pdist(input_matrix, 'euclidean'))
    number_of_row = distance_matrix.shape[0]

    # Initializing step (or step 2)
    # selected_index_list: list to track indexes that are selected (or I in the paper)
    # at the end its length should be number_of_row
    selected_index_list = [0]

    # unselected_index_list: list to track indexes that are not selected yet (or J in the paper)
    # at the end it should be empty
    unselected_index_list = np.arange(number_of_row)
    # initialize step
    unselected_index_list = np.delete(unselected_index_list, 0)
    # order: list that track the order of the final matrix (or R in the paper)
    order = [0]

    # Iteration (step 3)
    for r in range(1, number_of_row):
        i = selected_index_list[-1]
        d = distance_matrix[i, unselected_index_list]
        j = unselected_index_list[np.argmin(d)]
        selected_index_list.append(j)
        unselected_index_list = np.delete(unselected_index_list, np.where(unselected_index_list == j))
        order.append(j)

    # Return order dissimilarity matrix (step 4)
    return distance_matrix[np.ix_(order, order)]
    
def save_map(map_data, fig_name='vat-reorder.png', fig_type='png'):
    # Write figure out
    fig, ax = plt.subplots()
    ax.imshow(map_data, cmap='gray')
    ax.set_title('VAT Reordered Distance Matrix')
    fig.savefig(fig_name, format=fig_type)