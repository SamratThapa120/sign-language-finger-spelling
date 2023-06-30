import numpy as np

def interp1d_(x, target_len, method='random'):
    target_len = max(1,target_len)
    if method == 'random':
        if np.random.uniform() < 0.33:
            x = np.resize(x, (target_len,x.shape[1]))
        else:
            if np.random.uniform() < 0.5:
                x = np.resize(x, (target_len,x.shape[1]))
            else:
                x = np.resize(x, (target_len,x.shape[1]))
    else:
        x = np.resize(x, (target_len,x.shape[1]))
    return x

def flip_lr(x,CFG):
    x,y,z = np.moveaxis(x, -1, 0)
    x = 1-x
    new_x = np.stack([x,y,z], -1)
    new_x = new_x.transpose((1,0,2))
    lhand = new_x[CFG.LHAND, :, :]
    rhand = new_x[CFG.RHAND, :, :]
    new_x[CFG.LHAND, :, :] = rhand
    new_x[CFG.RHAND, :, :] = lhand
    # similar operations for llip, rlip, lpose, rpose, leye, reye, lnose, rnose
    new_x = new_x.transpose((1,0,2))
    return new_x

def spatial_random_affine(xyz,
    scale  = (0.8,1.2),
    shear = (-0.15,0.15),
    shift  = (-0.1,0.1),
    degree = (-30,30),
):
    center = np.array([0.5,0.5])
    if scale is not None:
        scale = np.random.uniform(*scale)
        xyz = scale*xyz

    if shear is not None:
        xy = xyz[:,:2]
        z = xyz[:,2:]
        shear_x = shear_y = np.random.uniform(*shear)
        if np.random.uniform() < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = np.array([
            [1.,shear_x],
            [shear_y,1.]
        ])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
        xyz = np.concatenate([xy,z], axis=-1)

    if degree is not None:
        xy = xyz[:,:2]
        z = xyz[:,2:]
        xy -= center
        degree = np.random.uniform(*degree)
        radian = degree/180*np.pi
        c = np.cos(radian)
        s = np.sin(radian)
        rotate_mat = np.array([
            [c,s],
            [-s, c],
        ])
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = np.concatenate([xy,z], axis=-1)

    if shift is not None:
        shift = np.random.uniform(*shift)
        xyz = xyz + shift

    return xyz

def temporal_crop(x, length):
    l = x.shape[0]
    offset = np.random.randint(0, max(1, l-length))
    x = x[offset:offset+length]
    return x


def temporal_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    l = x.shape[0]
    mask_size = np.random.uniform(*size)
    mask_size = int(l * mask_size)
    mask_offset = np.random.randint(0, max(1, l - mask_size))
    x[mask_offset:mask_offset+mask_size] = mask_value
    return x

def spatial_mask(x, size=(0.2,0.4), mask_value=float('nan')):
    mask_offset_y = np.random.uniform()
    mask_offset_x = np.random.uniform()
    mask_size = np.random.uniform(*size)
    mask_x = (mask_offset_x < x[...,0]) & (x[...,0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[...,1]) & (x[...,1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x[mask] = mask_value
    return x

def resample(x, rate=(0.8,1.2)):
    rate = np.random.uniform(rate[0], rate[1])
    length = x.shape[0]
    new_size = int(rate * length)
    new_x = interp1d_(x, new_size)
    return new_x

def augment_fn(x, always=False, max_len=None):
    if np.random.uniform() < 0.8 or always:
        x = resample(x, (0.5, 1.5))
    if np.random.uniform() < 0.5 or always:
        x = flip_lr(x, CFG)  # ensure that CFG is defined before calling this function
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if np.random.uniform() < 0.75 or always:
        x = spatial_random_affine(x)
    if np.random.uniform() < 0.5 or always:
        x = temporal_mask(x)
    if np.random.uniform() < 0.5 or always:
        x = spatial_mask(x)
    return x
