import numpy as np


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    d = 0
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 2]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 1] + tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind + 2] = tensor[..., ind + 2] - tensor[..., ind] + d  # Set w
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 1] + d  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0  # Set xmin
        tensor1[..., ind + 1] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind + 2] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0  # Set ymax
    return tensor1

