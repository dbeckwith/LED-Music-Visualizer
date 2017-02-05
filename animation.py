# -*- coding: utf-8 -*-

import numpy as np
import pyqtgraph as pg

import config
import util
from gui import gui


class ExpFilter:
    """Simple exponential smoothing filter"""
    def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        """Small rise / decay factors = more smoothing"""
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            pos = alpha > 0.0
            alpha[pos] = self.alpha_rise
            alpha[np.logical_not(pos)] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

_smoothing_filters = {}
def smooth(alpha_decay=0.5, alpha_rise=0.5):
    def decorator(f):
        def new_f(*args, **kwargs):
            val = f(*args, **kwargs)
            if f not in _smoothing_filters:
                _smoothing_filters[f] = ExpFilter(val, alpha_decay, alpha_rise)
                return val
            else:
                return _smoothing_filters[f].update(val)
        return new_f
    return decorator

class Animation(object):
    def __init__(self, audio_samples, sample_rate):
        self.sample_rate = sample_rate
        self.sample_count = audio_samples.shape[0]

        util.timer('Creating animation')
        self.spec = _make_spectrogram(audio_samples, sample_rate, int(config.PIXEL_COUNT / 2 * config.CHANNELS_PER_PIXEL))
        low, mid, hi = tuple(self._split_spec(self.spec, config.PIXEL_COUNT // 2))
        # TODO: map to other hues besides RGB? (might need to be linearly independent)
        self.frames = np.zeros((self.spec.shape[0], config.PIXEL_COUNT, config.CHANNELS_PER_PIXEL), dtype=np.float64)
        self.frames[:, :, 0] = np.concatenate((low[:, ::-1], low), axis=1)
        self.frames[:, :, 1] = np.concatenate((mid[:, ::-1], mid), axis=1)
        self.frames[:, :, 2] = np.concatenate((hi[:, ::-1], hi), axis=1)

        # TODO: maybe could do some kind of peak analysis to get specific instruments?

    def _split_spec(self, spec, n):
        ranges = np.linspace(0, spec.shape[1], config.CHANNELS_PER_PIXEL + 1, dtype=np.int_)
        for i in range(config.CHANNELS_PER_PIXEL):
            spec_from = ranges[i]
            spec_to = ranges[i + 1]
            spec_range = spec_to - spec_from
            channel_vals = np.empty((spec.shape[0], n), dtype=spec.dtype)
            for frame in range(spec.shape[0]):
                channel_vals[frame] = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, spec_to - spec_from), spec[frame, spec_from:spec_to])
            yield channel_vals

    @smooth(alpha_decay=0.2, alpha_rise=0.99)
    def get_frame(self, t):
        return _frame_interp(self.frames, self.sample_rate * self.frames.shape[0] / self.sample_count, t)

    @smooth(alpha_decay=0.2, alpha_rise=0.99)
    def get_spec_frame(self, t):
        return _frame_interp(self.spec, self.sample_rate * self.spec.shape[0] / self.sample_count, t)

def _frame_interp(frames, frame_rate, t):
    i = t * frame_rate
    if i <= 0:
        return frames[0]
    if i >= frames.shape[0] - 1:
        return frames[-1]
    i1 = np.floor(i)
    i2 = np.ceil(i)
    if i1 == i2:
        return frames[int(i1)]
    else:
        blend = i - i1
        return frames[int(i1)] * (1 - blend) + frames[int(i2)] * blend

def _make_spectrogram(samples, sample_rate, spectrogram_width):
    frame_size = int(20e-3 * sample_rate)
    frame_step = frame_size // 2

    samples = samples.copy()
    samples = np.mean(samples, axis=1)

    frame_count = int(np.ceil(len(samples) / frame_step))

    # pad samples to fit last frame
    pad_samples = (frame_count - 1) * frame_step + frame_size - len(samples)
    if pad_samples:
        samples = np.concatenate((samples, np.zeros(pad_samples)))

    frames = np.empty((frame_count, frame_size), dtype=samples.dtype)
    for frame_idx in range(frame_count):
        sample_idx = frame_idx * frame_step
        frames[frame_idx] = samples[sample_idx : sample_idx + frame_size]

    window = np.hanning(frame_size)
    frames *= window

    dft_size = 1024
    dft = np.fft.rfft(frames, n=dft_size)
    power_spectrum = np.square(np.abs(dft)) / frame_size
    spectrum_freqs = np.fft.rfftfreq(dft_size, d=1 / sample_rate)

    power_spectrum = _mel_filter(power_spectrum, spectrum_freqs, spectrogram_width)
    # power_spectrum = np.log(power_spectrum+1)

    print(np.min(power_spectrum), np.max(power_spectrum))
    power_spectrum = np.log(power_spectrum + 1)
    power_spectrum /= np.max(power_spectrum)

    # import matplotlib.pyplot as plt
    # plt.imshow(power_spectrum[10000:11000].T, cmap='gray')
    # plt.show()

    # exit()

    gui.debug_layout.addViewBox(
        row=1,
        col=0
    ).addItem(pg.ImageItem(
        image=power_spectrum
    ))

    return power_spectrum

def _mel_filter(power_spectrum, spectrum_freqs, num_filters):
    def freq_to_mel(f): return 1125 * np.log(1 + f / 700)
    def mel_to_freq(m): return 700 * (np.exp(m / 1125) - 1)

    print(power_spectrum.shape)

    spec_size = power_spectrum.shape[1]

    min_freq = 20
    # TODO: max freq should be some upper limit, or Nyquist freq?
    max_freq = 8000#self.sample_rate // 2

    min_freq = freq_to_mel(min_freq)
    max_freq = freq_to_mel(max_freq)

    filter_freqs = np.linspace(min_freq, max_freq, num_filters + 2)
    filter_freqs = mel_to_freq(filter_freqs)
    # print(filter_freqs)
    # print(spectrum_freqs)
    # TODO: this rebinning with the filters could be more accurate?
    # this round really impacts lower frequency bins
    filter_freqs = np.round(np.interp(filter_freqs, spectrum_freqs, np.arange(spec_size))).astype(np.int_)
    # print(filter_freqs)

    filterbanks = np.zeros((spec_size, num_filters), dtype=power_spectrum.dtype)
    filterbank_plot = gui.debug_layout.addPlot(
        row=0,
        col=0,
        title='Mel Filterbanks',
        labels={'left': 'Coefficient', 'bottom': 'Frequency'}
    )
    # import matplotlib.pyplot as plt
    for i in range(num_filters):
        filter_min = filter_freqs[i]
        filter_mid = filter_freqs[i + 1]
        filter_max = filter_freqs[i + 2]
        filterbanks[filter_min : filter_mid, i] = np.linspace(0, 1, filter_mid - filter_min)
        filterbanks[filter_mid - 1 : filter_max, i] = np.linspace(1, 0, filter_max - filter_mid + 1)
        filterbanks[:, i] /= ((filter_max - filter_min) / 2)
        # if i < 5:
        #     print(filter_min, filter_mid, filter_max)
        #     print(filterbanks[:, i])
        filterbank_plot.plot(y=filterbanks[:, i])
        # plt.plot(filterbanks[:, i])
    # plt.show()

    # print(power_spectrum.shape)
    # print(filterbanks.shape)

    power_spectrum_filtered = np.dot(power_spectrum, filterbanks)

    # print(power_spectrum_filtered.shape)

    return power_spectrum_filtered
