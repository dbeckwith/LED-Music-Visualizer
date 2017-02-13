# -*- coding: utf-8 -*-

import datetime
import numpy as np


_timer = None
_start_time = None
_timer_stage = None

def timer(stage=None):
    global _timer, _start_time, _timer_stage
    t = datetime.datetime.now()
    if _start_time == None:
        _start_time = t
    if _timer != None:
        print('{} done in {}'.format(_timer_stage, t - _timer))
    if stage is None:
        print('Done in {}'.format(t - _start_time))
    else:
        _timer = t
        _timer_stage = stage
        print('{}...'.format(stage))

def lerp(x, old_min, old_max, new_min, new_max, clip=False):
    new_x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    if clip:
        if new_min > new_max:
            new_min, new_max = new_max, new_min
        return np.clip(new_x, new_min, new_max)
    return new_x

def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    if output is None:
        output = np.zeros_like(input)

    sl = [slice(None)] * input.ndim

    filter_width = int(truncate * np.abs(sigma) + 0.5)
    filter = np.linspace(-filter_width, filter_width, filter_width * 2 + 1)
    filter = np.exp(-np.square(filter) / (2 * np.square(sigma))) / (sigma * np.sqrt(2 * np.pi))
    filter_sl = [np.newaxis] * input.ndim
    filter_sl[axis] = slice(None)
    filter = filter.__getitem__(filter_sl)

    in_sl = list(sl)
    out_sl = list(sl)
    axis_len = input.shape[axis]
    for i in range(axis_len):
        from_idx = i - filter_width
        from_extra = 0
        if from_idx < 0:
            from_extra = 0 - from_idx
            from_idx = 0

        to_idx = i + filter_width
        to_extra = 0
        if to_idx >= axis_len:
            to_extra = to_idx - (axis_len - 1)
            to_idx = axis_len - 1

        in_sl[axis] = slice(from_idx, to_idx + 1)
        x = input.__getitem__(in_sl)

        if from_extra > 0:
            from_extra_shape = list(input.shape)
            from_extra_shape[axis] = from_extra
            if mode == 'reflect':
                from_extra_sl = list(in_sl)
                from_extra_sl[axis] = slice(None, from_extra)
                from_extra = input.__getitem__(from_extra_sl)
                from_extra_sl[axis] = slice(None, None, -1)
                from_extra = from_extra.__getitem__(from_extra_sl)
            elif mode == 'constant':
                from_extra = np.tile(cval, from_extra_shape)
            elif mode == 'nearest':
                raise NotImplementedError
                from_extra_sl = list(in_sl)
                from_extra_sl[axis] = 0
                from_extra = np.tile(input.__getitem__(from_extra_sl), from_extra_shape)
            elif mode == 'mirror':
                raise NotImplementedError
            elif mode == 'wrap':
                raise NotImplementedError
            x = np.concatenate((from_extra, x), axis=axis)

        if to_extra > 0:
            to_extra_shape = list(input.shape)
            to_extra_shape[axis] = to_extra
            if mode == 'reflect':
                to_extra_sl = list(in_sl)
                to_extra_sl[axis] = slice(-to_extra, None)
                to_extra = input.__getitem__(to_extra_sl)
                to_extra_sl[axis] = slice(None, None, -1)
                to_extra = to_extra.__getitem__(to_extra_sl)
            elif mode == 'constant':
                to_extra = np.tile(cval, to_extra_shape)
            elif mode == 'nearest':
                raise NotImplementedError
                to_extra_sl = list(in_sl)
                to_extra_sl[axis] = -1
                to_extra = np.tile(input.__getitem__(to_extra_sl), to_extra_shape)
            elif mode == 'mirror':
                raise NotImplementedError
            elif mode == 'wrap':
                raise NotImplementedError
            x = np.concatenate((x, to_extra), axis=axis)

        out_sl[axis] = i
        val = x * filter
        val = np.sum(val, axis=axis)
        val += output.__getitem__(out_sl)
        output.__setitem__(out_sl, val)

    return output

if __name__ == '__main__':
    from scipy.ndimage.filters import gaussian_filter1d as sp_gaussian_filter1d
    x = np.zeros((2, 15, 2), dtype=np.float64)
    for i in range(3):
        x[:, i, 0] = x[:, -i-1, 0] = 1
        x[:, i, 1] = x[:, -i-1, 1] = 2
    f1 = sp_gaussian_filter1d(x, 1, axis=1)
    f2 = gaussian_filter1d(x, 1, axis=1)
    print(x)
    # print(f1)
    # print(f2)
    diff = f2 - f1
    error = np.max(np.abs(f2 - f1))
    print(diff)
    print(error)
    assert error < 1e-5
