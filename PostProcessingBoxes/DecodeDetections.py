import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec


class DecodeDetections(Layer):
    def __init__(self, confidence_thresh=0.01, iou_threshold=0.45, top_k=200, nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True, img_height=None, img_width=None, **kwargs):
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape)

    def call(self, y_pred, mask=None):
        cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[
            ..., -8]  # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[
            ..., -7]  # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(y_pred[..., -10] * y_pred[..., -2]) * y_pred[..., -6]  # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_pred[..., -9] * y_pred[..., -1]) * y_pred[..., -5]  # h = exp(h_pred * variance_h) * h_anchor

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1

        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), \
                   tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

        y_pred = tf.concat(values=[y_pred[..., :-12], xmin, ymin, xmax, ymax], axis=-1)

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[1]
        n_classes = y_pred.shape[2] - 4
        class_indices = tf.range(1, n_classes)

        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):
            def filter_single_class(index):
                confidences = tf.expand_dims(batch_item[..., index], axis=-1)
                class_id = tf.fill(dims=tf.shape(confidences), value=tf.cast(index, tf.float32))
                box_coordinates = batch_item[..., -4:]

                single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)

                threshold_met = single_class[:, 1] > self.tf_confidence_thresh
                single_class = tf.boolean_mask(tensor=single_class,
                                               mask=threshold_met)

                def perform_nms():
                    scores = single_class[..., 1]

                    xmin = tf.expand_dims(single_class[..., -4], axis=-1)
                    ymin = tf.expand_dims(single_class[..., -3], axis=-1)
                    xmax = tf.expand_dims(single_class[..., -2], axis=-1)
                    ymax = tf.expand_dims(single_class[..., -1], axis=-1)
                    boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                    maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                  scores=scores,
                                                                  max_output_size=self.tf_nms_max_output_size,
                                                                  iou_threshold=self.iou_threshold,
                                                                  name='non_maximum_suppresion')
                    maxima = tf.gather(params=single_class,
                                       indices=maxima_indices,
                                       axis=0)
                    return maxima

                def no_confident_predictions():
                    return tf.constant(value=0.0, shape=(1, 6))

                single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

                padded_single_class = tf.pad(tensor=single_class_nms,
                                             paddings=[[0, self.tf_nms_max_output_size - tf.shape(single_class_nms)[0]],
                                                       [0, 0]],
                                             mode='CONSTANT',
                                             constant_values=0.0)

                return padded_single_class

            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                                elems=tf.range(1, n_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128,
                                                back_prop=False,
                                                swap_memory=False,
                                                infer_shape=True,
                                                name='loop_over_classes')

            filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

            def top_k():
                return tf.gather(params=filtered_predictions,
                                 indices=tf.nn.top_k(filtered_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=filtered_predictions,
                                            paddings=[[0, self.tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], self.tf_top_k), top_k,
                                  pad_and_top_k)

            return top_k_boxes

        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor
