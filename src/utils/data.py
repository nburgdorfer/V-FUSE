import torch
import re
import numpy as np
import collections.abc as container_abcs

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')
_use_shared_memory = False
np_str_obj_array_pattern = re.compile(r'[SaUO]')
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def to_device(data, device):
    no_gpu_list = ["scene", "ref_index", "filenames", "num_frame"]

    for key,val in data.items():
        if (key not in no_gpu_list):
            data[key] = val.cuda(device, non_blocking=True)


def v(var, device, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var).float(), volatile=volatile)
    if cuda:
        res = res.cuda(device)
    return res


def npy(var):
    return var.data.cpu().numpy()

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type ==  'OpenGL':
        dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
    elif type == 'OpenCV':
        dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d

def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(batch, list) and isinstance(elem, tuple):
        #data = torch.cat((x[0] for x in batch))
        return [x[0] for x in batch]
    if type(elem) == tuple and elem[1] == 'varlen':
        return [x[0] for x in batch]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except:
            import ipdb
            ipdb.set_trace()

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    #elif isinstance(elem, int_classes):
    elif isinstance(elem, int):
        return torch.tensor(batch)
    #elif isinstance(elem, string_classes):
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def default_collatev1_1(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(batch[0])
    if isinstance(batch, list) and isinstance(elem, tuple):
        #data = torch.cat((x[0] for x in batch))
        return [x[0] for x in batch]
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except:
            import ipdb
            ipdb.set_trace()
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collatev1_1(
                [torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    #elif isinstance(batch[0], int_classes):
    elif isinstance(batch[0], int):
        return torch.tensor(batch)
    #elif isinstance(batch[0], string_classes):
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {
            key: default_collatev1_1([d[key] for d in batch])
            for key in batch[0]
        }
    elif isinstance(batch[0], tuple) and hasattr(batch[0],
                                                 '_fields'):  # namedtuple
        return type(batch[0])(*(default_collatev1_1(samples)
                                for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collatev1_1(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))
