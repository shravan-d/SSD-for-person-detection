import numpy as np


def sample_tensors(weights_list, sampling_instructions, axes):
    first_tensor = weights_list[0]
    out_shape = []  # Store the shape of the output tensor here.
    subsampled_weights_list = []  # Tensors after sub-sampling, but before up-sampling (if any).

    sampling_slices = []
    for i, sampling_inst in enumerate(sampling_instructions):
        if isinstance(sampling_inst, (list, tuple)):
            sampling_slices.append(np.array(sampling_inst))
            out_shape.append(len(sampling_inst))
        elif isinstance(sampling_inst, int):
            out_shape.append(sampling_inst)
            if sampling_inst == first_tensor.shape[i]:
                sampling_slice = np.arange(sampling_inst)
                sampling_slices.append(sampling_slice)
            elif sampling_inst < first_tensor.shape[i]:
                sampling_slice1 = np.array([0])
                sampling_slice2 = np.sort(np.random.choice(np.arange(1, first_tensor.shape[i]), sampling_inst - 1,
                                                           replace=False))
                sampling_slice = np.concatenate([sampling_slice1, sampling_slice2])
                sampling_slices.append(sampling_slice)
            else:
                print("Upsample no requirement")
        else:
            raise ValueError("Received `{}`".format(type(sampling_inst)))

    subsampled_first_tensor = np.copy(first_tensor[np.ix_(*sampling_slices)])
    subsampled_weights_list.append(subsampled_first_tensor)

    if len(weights_list) > 1:
        for j in range(1, len(weights_list)):
            this_sampling_slices = [sampling_slices[i] for i in axes[j - 1]]
            subsampled_weights_list.append(np.copy(weights_list[j][np.ix_(*this_sampling_slices)]))

    return subsampled_weights_list
