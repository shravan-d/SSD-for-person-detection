import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from AnchorBoxes.ConvertCoordinates import convert_coordinates


class AnchorBoxes(Layer):
    def __init__(self, img_height, img_width, this_scale, next_scale, aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True, this_steps=None,
                 this_offsets=None, clip_boxes=False, variances=[0.1, 0.1, 0.2, 0.2], coords='centroids',
                 normalize_coords=False, **kwargs):

        variances = np.array(variances)
        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        size = min(self.img_height, self.img_width)
        wh_list = []
        # concert aspect ratio to length and width of boxes
        for ar in self.aspect_ratios:
            if ar == 1:
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
            step_height = self.this_steps[0]
            step_width = self.this_steps[1]
        elif isinstance(self.this_steps, (int, float)):
            step_height = self.this_steps
            step_width = self.this_steps

        if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
            offset_height = self.this_offsets[0]
            offset_width = self.this_offsets[1]
        elif isinstance(self.this_offsets, (int, float)):
            offset_height = self.this_offsets
            offset_width = self.this_offsets

        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height,
                         feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width,
                         feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids',
                                           border_pixels='half')

        variances_tensor = np.zeros_like(
            boxes_tensor)  # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances  # Long live broadcasting
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor
