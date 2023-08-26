import tensorflow as tf
import numpy as np

def interp1d_(x, target_len, method='random'):
    target_len = tf.maximum(1,target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'bicubic')
            else:
                x = tf.image.resize(x, (target_len,tf.shape(x)[1]),'nearest')
    else:
        x = tf.image.resize(x, (target_len,tf.shape(x)[1]),method)
    return x

@tf.function()
def flip_lr(x,CFG):
    x,y,z = tf.unstack(x, axis=-1)
    x = 1-x
    new_x = tf.stack([x,y,z], -1)
    new_x = tf.transpose(new_x, [1,0,2])
    lhand = tf.gather(new_x, CFG.LHAND, axis=0)
    rhand = tf.gather(new_x, CFG.RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.LHAND)[...,None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.RHAND)[...,None], lhand)
    llip = tf.gather(new_x, CFG.LLIP, axis=0)
    rlip = tf.gather(new_x, CFG.RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.LLIP)[...,None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.RLIP)[...,None], llip)
    lpose = tf.gather(new_x, CFG.LPOSE, axis=0)
    rpose = tf.gather(new_x, CFG.RPOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.LPOSE)[...,None], rpose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.RPOSE)[...,None], lpose)
    leye = tf.gather(new_x, CFG.LEYE, axis=0)
    reye = tf.gather(new_x, CFG.REYE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.LEYE)[...,None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.REYE)[...,None], leye)
    lnose = tf.gather(new_x, CFG.LNOSE, axis=0)
    rnose = tf.gather(new_x, CFG.RNOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.LNOSE)[...,None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(CFG.RNOSE)[...,None], lnose)
    new_x = tf.transpose(new_x, [1,0,2])
    return new_x


@tf.function()
def spatial_random_affine(xyz,
    scale  = (0.5,2),
    shear = (-0.3,-0.3),
    shift  = (-0.5,0.5),
    degree = (-45,45),
):
    center = tf.constant([0.5,0.5])
    if scale is not None:
        scale = tf.random.uniform((),*scale)
        xyz = scale*xyz

    if shear is not None:
        xy = xyz[...,:2]
        z = xyz[...,2:]
        shear_x = shear_y = tf.random.uniform((),*shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = tf.identity([
            [1.,shear_x],
            [shear_y,1.]
        ])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
        xyz = tf.concat([xy,z], axis=-1)

    if degree is not None:
        xy = xyz[...,:2]
        z = xyz[...,2:]
        xy -= center
        degree = tf.random.uniform((),*degree)
        radian = degree/180*np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([
            [c,s],
            [-s, c],
        ])
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = tf.concat([xy,z], axis=-1)

    if shift is not None:
        shift = tf.random.uniform((),*shift)
        xyz = xyz + shift

    return xyz

@tf.function()
def spatial_random_affine_perframe(xyz,
    scale  = (0.5,2),
    shear = (-0.3,-0.3),
    shift  = (-0.5,0.5),
    degree = (-45,45),
):
    seqlen = tf.shape(xyz)[0]
    center = tf.constant([0.5,0.5])
    if scale is not None:
        scale = tf.random.uniform((seqlen,1,1),*scale)        
        xyz = scale*xyz

    if shear is not None:
        xy = xyz[...,:2]
        z = xyz[...,2:]
        shear_x = tf.random.uniform((seqlen, 1, 1), *shear)
        shear_y = tf.random.uniform((seqlen, 1, 1), *shear)
        mask = tf.random.uniform((seqlen, 1, 1)) < 0.5
        shear_x = tf.where(mask, shear_x, 0.)
        shear_y = tf.where(mask, 0., shear_y)

        ones = tf.ones_like(shear_x)
        shear_mat = tf.reshape(tf.concat([
            tf.concat([ones, shear_x], axis=2),
            tf.concat([shear_y, ones], axis=2)
        ], axis=1), [seqlen, 2, 2])

        xy = tf.matmul(xy, shear_mat)
        center = center + tf.stack([shear_y[..., 0], shear_x[..., 0]], axis=-1)
        xyz = tf.concat([xy, z], axis=-1)    
    if degree is not None:
        xy = xyz[...,:2] - center
        z = xyz[...,2:]
  
        degree = tf.random.uniform((seqlen, 1), *degree)
        radian = degree / 180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)

        rotate_mat = tf.reshape(tf.stack([
            tf.stack([c, s], axis=-1),
            tf.stack([-s, c], axis=-1)
        ], axis=1), [seqlen, 2, 2])

        xy = tf.matmul(xy, rotate_mat)
        xy += center
        xyz = tf.concat([xy, z], axis=-1)

    if shift is not None:
        shift = tf.random.uniform((seqlen, 1, 1), *shift)
        xyz += shift

    return xyz
    
def temporal_crop(x, length):
    l = tf.shape(x)[0]
    offset = tf.random.uniform((), 0, tf.clip_by_value(l-length,1,length), dtype=tf.int32)
    x = x[offset:offset+length]
    return x

def temporal_mask(x, targetlen,size=(0.2,0.4)):
    shape = tf.shape(x)
    p = tf.random.uniform([], minval=size[0], maxval=size[1])
    logits = tf.stack([tf.zeros(shape[0]), tf.ones(shape[0]) * tf.math.log(1.0 - p) - tf.math.log(p)], axis=-1)
    mask = tf.random.categorical(logits, 1)
    mask = tf.reshape(tf.cast(mask, tf.bool), [-1])
    masked_x = tf.boolean_mask(x, mask, axis=0)
    if tf.shape(masked_x)[0] < 2*targetlen:
        return x
    else:
        return masked_x


def spatial_mask(x, erasable_landmarks, size=(0,1), mask_value=float('nan')):
    p = tf.random.uniform((), minval=size[0], maxval=size[1])

    # Use tf.shape to get dynamic shape
    mask_shape = [tf.shape(x)[0], tf.size(erasable_landmarks), tf.shape(x)[2]]

    mask = tf.random.uniform(mask_shape) < p
    mask = tf.reduce_any(mask, axis=-1, keepdims=True)
    mask = tf.repeat(mask, tf.shape(x)[-1], axis=-1)
    update = tf.fill(mask_shape, mask_value)

    indices = tf.meshgrid(tf.range(tf.shape(x)[0]), erasable_landmarks, tf.range(tf.shape(x)[2]), indexing='ij')
    indices = tf.stack(indices, axis=-1)

    masked = tf.tensor_scatter_nd_update(x, indices, tf.where(mask, update, tf.gather_nd(x, indices)))
    return masked

def scale_hands(x, left_hand,right_hand, scale=(0.5,2)):
    scale_factor = tf.random.uniform((), *scale)
    for hland,centerpoint in zip([left_hand,right_hand],[504,505]):
        hands = tf.gather(x, hland, axis=1)
        ref_point = tf.gather(x, [centerpoint], axis=1)
        centered_hands = hands - ref_point
        scaled_hands = centered_hands * scale_factor
        scaled_hands += ref_point
        frames = tf.shape(x)[0]
        frames_indices = tf.range(frames)
        hand_landmarks_indices = tf.constant(hland)
        grid_frames, grid_landmarks = tf.meshgrid(frames_indices, hand_landmarks_indices, indexing='ij')
        combined_indices = tf.stack([grid_frames, grid_landmarks], axis=-1)
        x = tf.tensor_scatter_nd_update(x, tf.reshape(combined_indices, [-1, 2]), tf.reshape(scaled_hands, [-1, 3]))
    return x


def resample(x, rate=(0.8,1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate*tf.cast(length,tf.float32), tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x


@tf.function()
def freeze_frames(x, maxlen):
    seqlen = tf.shape(x)[0]
    if seqlen >= maxlen:
        return x
    else:
        # Compute the number of repeats
        repeats_needed = tf.cast(maxlen - seqlen,tf.float32)

        # Generate random numbers, apply softmax, and scale
        random_numbers = tf.random.uniform(shape=(seqlen,), minval=0, maxval=1)
        softmax_numbers = tf.nn.softmax(random_numbers)
        repeats = 1+tf.cast(softmax_numbers*repeats_needed, tf.float32)
        x = tf.repeat(x, repeats=tf.cast(repeats, tf.int32), axis=0)
        return x

@tf.function()
def augment_fn(x, tarlen,CFG):
    if tf.random.uniform(())<CFG.tempmask_probability:
        x = temporal_mask(x,tarlen,CFG.tempmask_range)
    if tf.random.uniform(())<CFG.handonly_scale_probability:
        x = scale_hands(x,CFG.LHAND,CFG.RHAND)
    if tf.random.uniform(())<CFG.freeze_probability:
        x = freeze_frames(x,CFG.max_len)
    if tf.random.uniform(())<CFG.flip_lr_probability:
        x = flip_lr(x,CFG)
    if tf.random.uniform(())<CFG.random_affine_probability:
        if tf.random.uniform(())<CFG.perframe_random_affine_proportion:
            x = spatial_random_affine_perframe(x)
        else:
            x = spatial_random_affine(x)
    if tf.random.uniform(())<CFG.erase_probability:
        x = spatial_mask(x,CFG.ERASABLE_LANDMARKS)
    return x
