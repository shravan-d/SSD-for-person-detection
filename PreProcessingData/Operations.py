import numpy as np


class ConvertTo3Channels:
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        if labels is None:
            return image
        else:
            return image, labels


class Resize:
    def __init__(self, height, width, interpolation_mode=cv2.INTER_LINEAR, box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):
        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image, dsize=(self.out_width, self.out_height), interpolation=self.interpolation_mode)
        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin + 1, ymax + 1]] = np.round(
                    labels[:, [ymin + 1, ymax + 1]] * (img_height / self.out_height), decimals=0)
                labels[:, [xmin + 1, xmax + 1]] = np.round(
                    labels[:, [xmin + 1, xmax + 1]] * (img_width / self.out_width), decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

class BoxFilter:
    def __init__(self,
                 check_overlap=True, check_min_area=True,  check_degenerate=True, overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0), min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},  border_pixels='half'):
      self.overlap_criterion = overlap_criterion
      self.overlap_bounds = overlap_bounds
      self.min_area = min_area
      self.check_overlap = check_overlap
      self.check_min_area = check_min_area
      self.check_degenerate = check_degenerate
      self.labels_format = labels_format
      self.border_pixels = border_pixels

    def __call__(self, labels, image_height=None, image_width=None):
        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:
            non_degenerate = (labels[:,xmax] > labels[:,xmin]) * (labels[:,ymax] > labels[:,ymin])
            requirements_met *= non_degenerate
        return labels[requirements_met]
