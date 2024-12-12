import functools
from typing import Mapping, Optional, List
import gin

import seqio
from seqio import utils

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

DEFAULT_EXTRA_IDS = 0
OutputFeaturesType = Mapping[str, utils.Feature]


def _append_to_innermost_axis(
    tensor: tf.Tensor, scalar: tf.Tensor,
) -> tf.Tensor:
  """Appends `scalar` to each slice in the innermost axis of `tensor`.

  >>> _append_to_innermost_axis([1, 2, 3], -1)
  [1, 2, 3, -1]
  >>> _append_to_innermost_axis([[1, 2], [3, 4]], -1)
  [[1, 2, -1], [3, 4, -1]]
  >>> _append_to_innermost_axis(tf.ragged.constant([[1, 2], [3]]), -1)
  [[1, 2, -1], [3, -1]]

  Args:
    tensor: The tensor that should have a value appended.
    scalar: The value to append.

  Returns:
    A copy of `tensor` with `scalar` appended to each slice along
    the innermost axis.
  """
  if isinstance(tensor, tf.RaggedTensor):
    if tensor.shape.rank > 2:
      return tensor.with_values(
          _append_to_innermost_axis(tensor.values, scalar)
      )
    else:
      return tf.concat([tensor, tf.fill([tensor.nrows(), 1], scalar)], axis=1)
  else:
    ndims = tf.rank(tensor)
    paddings = tf.concat(
        [tf.zeros((ndims - 1, 2), dtype=tf.int32), tf.constant([[0, 1]])],
        axis=0,
    )
    return tf.pad(tensor, paddings=paddings, constant_values=scalar)


def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping."""

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id


def make_autoregressive_inputs(
    targets: tf.Tensor,
    sequence_id: tf.Tensor = None,
    output_dtype: Optional[tf.dtypes.DType] = None,
    bos_id: int = 0,
) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.

  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.

  For the first element of each sequence, the returned input id is 0.

  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

  ```
        targets = [3, 8, 2, 9, 2, 5, 4, 2, -1, -1]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [1, 3, 8, 1, 9, 1, 5, 4, -1, -1]
                            |     |        |
                            These positions are set to 0 if sequence_id is not
                            None.
  ```

  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.
    output_dtype: an optional output data type.
    bos_id: bos id.

  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  output_dtype = output_dtype or targets.dtype
  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
        "The sequence_id should be integer-valued tensors for a packed dataset."
    )
  if sequence_id is not None and len(targets.shape) > 1:
    raise ValueError(
        "Only 1-D sequences are supported with packing. Got a "
        f"packed {len(targets.shape)}-D sequence."
    )

  inputs = _shift_right_by_one(targets, bos_id)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id, _shift_right_by_one(sequence_id)
    )
    not_first_in_sequence = tf.cast(not_first_in_sequence, output_dtype)
    first_ids = tf.cast((1 - not_first_in_sequence) * bos_id, output_dtype)
    inputs = inputs * not_first_in_sequence + first_ids
  return inputs


@tf.function
def sum_except_first_axis(tensor):
    # Compute the sum along all axes except the first
    axes_to_sum = tuple(range(1, len(tensor.shape)))
    return tf.reduce_sum(tensor, axis=axes_to_sum)


@seqio.map_over_dataset()
def add_segment_ids(ex):
    ex["subsegment_ids"] = tf.zeros_like(ex["target_tokens"], dtype=tf.int32)
    return ex


def trim_and_pad_dataset(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Trim and pad first dimension of features to `feature_lengths`.

  Args:
    dataset: tf.data.Dataset, the dataset to trim/pad examples in.
    feature_lengths: map from feature key to final length. Other features will
      be returned unchanged.

  Returns:
    Trimmed/padded tf.data.Dataset.
  """

  def _trim_and_pad(k: str, t: tf.Tensor) -> tf.Tensor:
    """Trim/pad to the first axis of `t` to be of size `length`."""
    if k not in feature_lengths:
      return t
    if isinstance(t, tf.RaggedTensor):
      t = t.to_tensor()

    constant_values = -1
    length_k = feature_lengths[k]
    if isinstance(length_k, int):
      t = t[:length_k]
      pad_amt = length_k - tf.shape(t)[0]
      padded_t = tf.pad(t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1), constant_values=constant_values)
      padded_t.set_shape([length_k] + t.shape.as_list()[1:])
      return padded_t

    slices = tuple((slice(0, limit) for limit in length_k))
    t = t[slices]
    pad_amt = tf.pad((length_k - tf.shape(t))[..., None], ((0, 0), (1, 0)), constant_values=constant_values)
    padded_t = tf.pad(t, pad_amt, constant_values=constant_values)
    padded_t.set_shape(length_k)
    return padded_t

  return dataset.map(
      lambda x: {k: _trim_and_pad(k, t) for k, t in x.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )


def get_3d_subsegments(segmented_suffix):
    q_lens, text_lens = segmented_suffix.nested_row_lengths()
    text_segments = tf.range(0, tf.shape(text_lens)[0], dtype=tf.int32)
    question_repeat = tf.reshape(tf.stack([tf.ones_like(q_lens), q_lens-1], 1), [-1])
    question_offset = tf.range(1, tf.shape(q_lens)[0]+1, dtype=tf.int32)*200
    question_offset = tf.reshape(tf.stack([question_offset, question_offset-100], 1), [-1])
    text_segments = text_segments + tf.repeat(question_offset, question_repeat)
    segment_ids = tf.cast(tf.repeat(text_segments, text_lens), tf.int32)
    return segment_ids


def assert_not_truncated(ds, keys, max_val):
    def _check(ex):
        for k in keys:
            tf.assert_less(tf.shape(ex[k])[0], max_val+1,
                           message=f"Field {k} was unexpectedly truncated max_len={max_val}")
        return ex
    return ds.map(_check)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def denormalize_boxes(boxes, image_shape):
    """Converts boxes normalized by [height, width] to pixel coordinates.
    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates of
        boxes in ymin, xmin, ymax, xmax order.
      image_shape: a list of two integers, a two-element vector or a tensor such
        that all but the last dimensions are `broadcastable` to `boxes`. The last
        dimension is 2, which represents [height, width].
    Returns:
      denormalized_boxes: a tensor whose shape is the same as `boxes` representing
        the denormalized boxes.
    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    with tf.name_scope('denormalize_boxes'):
      if isinstance(image_shape, list) or isinstance(image_shape, tuple):
        height, width = image_shape
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
      else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height, width = tf.split(image_shape, 2, axis=-1)

      ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
      ymin = ymin * height
      xmin = xmin * width
      ymax = ymax * height
      xmax = xmax * width

      denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
      return denormalized_boxes


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, value=0):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    return tf.pad(image, [
        [offset_height, after_padding_height],
        [offset_width, after_padding_width],
        [0, 0]
    ], constant_values=value)


def resize_and_crop_boxes(boxes, image_scale, output_size, offset, paddings):
    """Resizes boxes to output size with scale and offset.
    Args:
      boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
      image_scale: 2D float `Tensor` representing scale factors that apply to
        [height, width] of input image.
      output_size: 2D `Tensor` or `int` representing [height, width] of target
        output image size.
      offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
        boxes.
      paddings: 2D `Tensor` representing top/left paddings.
    Returns:
      boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
    """
    # Adjusts box coordinates based on image_scale, offset and paddings.
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    boxes += tf.tile(tf.expand_dims(paddings, axis=0), [1, 2])
    # Clips the boxes.
    boxes = clip_boxes(boxes, output_size)
    return boxes

def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.
  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.
  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height, width, height, width]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack(
          [height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes
  
  
def get_non_empty_box_indices(boxes):
    """Get indices for non-empty boxes."""
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(
        tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
    return indices[:, 0]


def resize_and_pad(image, desired_output_size, masks=None, boxes=None, labels=None,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True, boxes1=None, filter_box=True,
                   desired_target_size=None, random_scale_ratio=0.0,
                   resize_method=tf.image.ResizeMethod.BILINEAR, return_outputs=True,
                   pad_value=0, normalize=True):
    desired_height, desired_width = desired_output_size
    desired_height_f = tf.cast(desired_height, dtype=tf.float32)
    desired_width_f = tf.cast(desired_width, dtype=tf.float32)

    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

    if boxes is not None:
        # Converts boxes from normalized coordinates to pixel coordinates.
        # Now the coordinates of boxes are w.r.t. the original image.
        boxes = denormalize_boxes(boxes, [height, width])

    if boxes1 is not None:
        boxes1 = denormalize_boxes(boxes1, [height, width])

    if do_random_scale:
        random_scale_factor = tf.random.uniform([], random_scale_min, random_scale_max)
        if not shrink_both_sides:
            # Max random is where scale * W > W_desired
            #                     scale * H > H_desired
            rsf_max = tf.maximum(desired_width_f / width, desired_height_f / height)
            random_scale_factor = tf.minimum(rsf_max, random_scale_factor)

        scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
        scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width

        image_scale = tf.cond(tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(random_scale_ratio, tf.float32)),
            lambda: tf.maximum(image_scale_x, image_scale_y),
            lambda: tf.minimum(image_scale_x, image_scale_y))

        # image_scale = tf.minimum(image_scale_x, image_scale_y)

        # Conceptual captions has some REALLY WIDE images I believe
        # this ensures that we won't scale any side lower than to 64
        image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

        # Select non-zero random offset (x, y) if scaled image is larger than
        # self._output_size.
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - desired_height, tf.float32)
        offset_x = tf.cast(scaled_width - desired_width, tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
    else:
        image_scale_y = desired_height_f / height
        image_scale_x = desired_width_f / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.constant(0)
        offset_x = tf.constant(0)

    # Now resize and crop
    if resize_method == 'random' and do_random_scale:
        resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
        image = apply_with_random_selector(
            image,
            lambda x, method_idx: tf.image.resize(x, [scaled_height, scaled_width],
                                                  tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                                  antialias=True),
            num_cases=len(resize_methods))

    elif resize_method != 'random':
        image = tf.image.resize(image, [scaled_height, scaled_width], method=resize_method, antialias=True)
    else:
        image = tf.image.resize(image, [scaled_height, scaled_width],
                                method=tf.image.ResizeMethod.BILINEAR, antialias=True)

    image = tf.clip_by_value(image, 0.0, 1.0)

    # H x W x C
    image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]

    H = tf.shape(image)[0]
    W = tf.shape(image)[1]

    top_pad = (desired_height - H) // 2
    left_pad = (desired_width - W) // 2

    image_mask = pad_to_bounding_box(
        tf.ones_like(image, dtype=tf.bool), top_pad, left_pad, desired_height, desired_width)[:,:,0]

    image = pad_to_bounding_box(image, top_pad, left_pad, desired_height, desired_width, value=pad_value)

    if isinstance(desired_height, int) and isinstance(desired_width, int):
        image.set_shape([desired_height, desired_width, 3])

    if masks is not None and tf.size(masks) != 0:
        masks = tf.image.resize(masks, [scaled_height, scaled_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if len(masks.shape) == 3:
            masks = masks[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]
        else:
            masks = masks[:, offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]

        masks = pad_to_bounding_box(masks, top_pad, left_pad, desired_height, desired_width)
        masks = tf.image.resize(masks, desired_target_size,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    indices = None
    if boxes is not None:
        # assert ValueError("the box need to be shift which is not tested yet.")
        boxes = resize_and_crop_boxes(
            boxes,
            tf.stack([image_scale, image_scale]),
            [desired_height, desired_width],
            tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32),
            tf.cast(tf.stack([top_pad, left_pad]), dtype=tf.float32))

        if filter_box:
            indices = get_non_empty_box_indices(boxes)
        else:
            indices = tf.range(tf.shape(boxes)[0])
        boxes = tf.gather(boxes, indices)

        if labels is not None:
            labels = tf.gather(labels, indices)

        if boxes1 is not None:
            boxes1 = resize_and_crop_boxes(
                boxes1,
                tf.stack([image_scale, image_scale]),
                [desired_height, desired_width],
                tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32),
                tf.cast(tf.stack([top_pad, left_pad]), dtype=tf.float32))

    image_info = tf.stack([
        tf.cast(top_pad, tf.float32),
        tf.cast(left_pad, tf.float32),
        1.0 / image_scale,
        height,
        width,
        tf.cast(offset_y, dtype=tf.float32) / height,
        tf.cast(offset_x, dtype=tf.float32) / width,
        tf.cast(offset_y, dtype=tf.float32),
        tf.cast(offset_x, dtype=tf.float32),
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32),
        ])

    if boxes1 is not None:
        outputs = (image_info, masks, boxes, labels, indices, boxes1)
    else:
        outputs = (image_info, masks, boxes, labels, indices)

    if normalize:
        image = normalize_image(image)

    if return_outputs:
        return image, image_mask, outputs
    else:
        return image, image_mask


def _remove_bars_from_frames(frames, black_bar=True, threshold=32, max_perc_to_trim=0.3):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim x% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars####################
    frames_shape = tf.shape(frames)
    h, w = frames_shape[1], frames_shape[2]
    if black_bar:
      has_content = tf.reduce_max(frames, axis=(0, -1)) >= threshold
    else:
      has_content = tf.reduce_min(frames, axis=(0, -1)) <= threshold

    y_frames = tf.cast(tf.reshape(tf.where(tf.reduce_any(has_content, axis=1)), [-1]), tf.int32)
    nhbars = tf.shape(y_frames)[0]
    y_frames = tf.cond(nhbars > 0, lambda: y_frames, lambda: tf.expand_dims(tf.cast(h // 2, tf.int32), axis=0))

    y1 = tf.minimum(y_frames[0], tf.cast(tf.cast(h, tf.float32) * max_perc_to_trim, tf.int32))
    y2 = tf.maximum(y_frames[-1] + 1, tf.cast(tf.cast(h, tf.float32) * (1 - max_perc_to_trim), tf.int32))

    x_frames = tf.cast(tf.reshape(tf.where(tf.reduce_any(has_content, axis=0)), [-1]), tf.int32)
    nvbars = tf.shape(x_frames)[0]
    x_frames = tf.cond(nvbars > 0, lambda: x_frames, lambda: tf.expand_dims(tf.cast(w // 2, tf.int32), axis=0))

    x1 = tf.minimum(x_frames[0], tf.cast(tf.cast(w, tf.float32) * max_perc_to_trim, tf.int32))
    x2 = tf.maximum(x_frames[-1] + 1, tf.cast(tf.cast(w, tf.float32) * (1 - max_perc_to_trim), tf.int32))

    frames = frames[:, y1:y2, x1:x2]
    return frames

def convert_video_dtype(video,dtype):
    """
    Converts tensor to dtype and scales the values. 
    Video equivalent of tf.convert_image_dtype: https://www.tensorflow.org/api_docs/python/tf/image/convert_image_dtype
    """
    return tf.map_fn(
        fn=functools.partial(
            tf.image.convert_image_dtype,
            dtype=dtype),
        elems=video,
        fn_output_signature=dtype)


def stateless_shuffle(x: tf.Tensor, seed):
  if hasattr(tf.random.experimental, 'stateless_shuffle'):
    return tf.random.experimental.stateless_shuffle(x, seed=seed)
  else:
    vals = tf.random.stateless_uniform(tf.shape(x)[:1], seed)
    ixs = tf.argsort(vals)
    return tf.gather(x, ixs)


def stateless_permutation(n: int, seed):
    if hasattr(tf.random.experimental, 'stateless_shuffle'):
        ix = tf.range(0, n, dtype=tf.int32)
        return tf.random.experimental.stateless_shuffle(ix, seed=seed)
    else:
        vals = tf.random.stateless_uniform(n, seed)
        return tf.argsort(vals)


@seqio.map_over_dataset
def _strip_metadata(example):
    return {k: v for k, v in example.items() if not k.startswith('metadata/')}


def sample_patches(mask, n_patches, stateless=False, seeds=None):
  input_sample_valid = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
  input_sample_masked = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask == 0)
  if stateless:
    encoder_pos_ids = tf.concat([
      stateless_shuffle(input_sample_valid, seeds[0]),
      stateless_shuffle(input_sample_masked, seeds[1])], axis=0)[:n_patches]
  else:
    encoder_pos_ids = tf.concat([
      tf.random.shuffle(input_sample_valid),
      tf.random.shuffle(input_sample_masked)], axis=0)[:n_patches]
  encoder_pos_ids = tf.reshape(encoder_pos_ids, (n_patches,))
  encoder_pos_ids = tf.cast(encoder_pos_ids, tf.int32)
  return encoder_pos_ids


@gin.configurable()
def normalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  """Normalizes the image to zero mean and unit variance."""
  offset = tf.constant(offset)
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= tf.cast(offset, image.dtype)

  scale = tf.constant(scale)
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= tf.cast(scale, image.dtype)
  return image


def unnormalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  """Normalizes the image to zero mean and unit variance."""
  scale = tf.cast(tf.expand_dims(tf.expand_dims(tf.constant(scale), axis=0), axis=0), image.dtype)
  image *= scale

  offset = tf.cast(tf.expand_dims(tf.expand_dims(tf.constant(offset), axis=0), axis=0), image.dtype)
  image += offset
  return image


def flatten_parts(ds: tf.data.Dataset, parts: List[str], add_index=False, dataset_size=None) -> tf.data.Dataset:
    def _flatten(ex):
        flat_key = {k: ex[k] for k in parts}
        if add_index:
            flat_key['index'] = tf.range(len(ex[parts[0]]))

        flat_ds = tf.data.Dataset.from_tensor_slices(flat_key)

        def _merge(_flat_ex):
            for k, v in ex.items():
                if k not in parts:
                    _flat_ex[k] = v
            return _flat_ex
        return flat_ds.map(_merge)

    ds = ds.flat_map(_flatten)
    if dataset_size is not None:
        ds = tf.data.experimental.assert_cardinality(dataset_size)(ds)
    return ds
