import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import numpy as np
import os
from PreProcessingData.Operations import *


class DataGenerator:

    def __init__(self, load_images_into_memory=False, hdf5_dataset_path=None, filenames=None, filenames_type='text',
                 images_dir=None, labels=None, image_ids=None, eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        self.labels_output_format = labels_output_format
        self.labels_format = {'class_id': labels_output_format.index('class_id'),
                              'xmin': labels_output_format.index('xmin'),
                              'ymin': labels_output_format.index('ymin'),
                              'xmax': labels_output_format.index('xmax'),
                              'ymax': labels_output_format.index('ymax')}  # This dictionary is for internal use.

        self.dataset_size = 0  # As long as we haven't loaded anything yet, the dataset size is zero.
        self.load_images_into_memory = load_images_into_memory
        self.images = None  # The only way that this list will not stay `None` is if `load_images_into_memory == True`.
        self.filenames = None
        self.labels = None
        self.image_ids = None
        self.eval_neutral = None
        self.hdf5_dataset = None

    def parse_csv(self, images_dir, labels_filename, input_format, include_classes='all', random_sample=False,
                  ret=False, verbose=True):
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes
        self.filenames = []
        self.image_ids = []
        self.labels = []
        data = []

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread)  # Skip the header row.
            for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
                if self.include_classes == 'all' or int(row[self.input_format.index(
                        'class_id')].strip()) in self.include_classes:  # If the class_id is among the classes that are to be included in the dataset...
                    box = []  # Store the box class and coordinates here
                    box.append(row[self.input_format.index(
                        'image_name')].strip())  # Select the image name column in the input format and append its content to `box`
                    for element in self.labels_output_format:  # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(
                            element)].strip()))  # ...select the respective column in the input format and append it to `box`.
                    data.append(box)

        data = sorted(data)
        current_file = data[0][0]  # The current image for which we're collecting the ground truth boxes
        current_image_id = data[0][0].split('.')[
            0]  # The image ID will be the portion of the image name before the first dot.
        current_labels = []  # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):
            if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            else:  # If this box belongs to a new image file
                self.labels.append(np.stack(current_labels, axis=0))
                self.filenames.append(os.path.join(self.images_dir, current_file))
                self.image_ids.append(current_image_id)

                current_labels = []  # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if ret:  # In case we want to return these
            return self.images, self.filenames, self.labels, self.image_ids

    def generate(self, batch_size=32, shuffle=False, transformations=[], label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False, degenerate_box_handling='remove'):
        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False, check_min_area=False, check_degenerate=True,
                                   labels_format=self.labels_format)
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format
        current = 0
        while True:
            batch_X, batch_y = [], []
            batch_indices = self.dataset_indices[current:current + batch_size]
            batch_filenames = self.filenames[current:current + batch_size]
            for filename in batch_filenames:
                with Image.open(filename + '.jpg') as image:
                    batch_X.append(np.array(image, dtype=np.uint8))
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current + batch_size])
            else:
                batch_y = None

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current + batch_size]
            else:
                batch_eval_neutral = None

            # Get the image IDs for this batch (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current + batch_size]
            else:
                batch_image_ids = None

            current += batch_size
            batch_items_to_remove = []  # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):
                if not (self.labels is None):
                    batch_y[i] = np.array(batch_y[i])
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue
                if transformations:
                    inverse_transforms = []
                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i],
                                                                                      return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[
                                i] is None:  # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and (
                                    'return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                if not (self.labels is None):
                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']
                    if np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or np.any(
                            batch_y[i][:, ymax] - batch_y[i][:, ymin] <= 0):
                        if degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            batch_X = np.array(batch_X)
            batch_X = np.array(batch_X)
            if batch_X.size == 0:
                raise DegenerateBatchError("You produced an empty batch.")
            if not (label_encoder is None or self.labels is None):
                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None
            else:
                batch_y_encoded = None
                batch_matched_anchors = None
            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)

            yield ret
