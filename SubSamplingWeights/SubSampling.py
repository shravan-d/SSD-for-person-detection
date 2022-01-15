import h5py
import shutil
import numpy as np
from SubSamplingWeights.SampleTensors import sample_tensors

weights_source_path = '/PretrainedWeights/VGG_coco_SSD_300x300_iter_400000.h5'
weights_destination_path = '/PretrainedWeights/VGG_coco_SSD_300x300_iter_400000.subsampled_2_classes.h5'

shutil.copy(weights_source_path, weights_destination_path)
weights_source_file = h5py.File(weights_source_path, "r")
weights_destination_file = h5py.File(weights_destination_path, "a")

classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']
conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights before sampling:".format(classifier_names[0]))
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

n_classes_source = 81
classes_of_interest = [0, 1]

for name in classifier_names:
    kernel = weights_source_file[name][name]['kernel:0'][()]
    bias = weights_source_file[name][name]['bias:0'][()]

    height, width, in_channels, out_channels = kernel.shape
    print(kernel.shape)
    subsampling_indices = []
    for i in range(int(out_channels / n_classes_source)):
        indices = np.array(classes_of_interest) + i * n_classes_source
        subsampling_indices.append(indices)
    subsampling_indices = list(np.concatenate(subsampling_indices))
    print(subsampling_indices)
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]])

    del weights_destination_file[name][name]['kernel:0']
    del weights_destination_file[name][name]['bias:0']
    weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
    weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

weights_destination_file.flush()

conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights after sampling:".format(classifier_names[0]))
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)
