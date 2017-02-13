# -*- coding: utf-8 -*-

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
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
def smooth(alpha_decay=0.5, alpha_rise=0.5, key=None):
    def decorator(f):
        if key is None:
            _key = f
        else:
            _key = key
        def new_f(*args, **kwargs):
            val = f(*args, **kwargs)
            if f not in _smoothing_filters:
                _smoothing_filters[_key] = ExpFilter(val, alpha_decay, alpha_rise)
                return val
            else:
                return _smoothing_filters[_key].update(val)
        return new_f
    return decorator

class Animation(object):
    def __init__(self, audio_samples, sample_rate):
        self.sample_rate = sample_rate
        self.sample_count = audio_samples.shape[0]
        self.duration = self.sample_count / self.sample_rate

        util.timer('Creating spectrogram')

        self.spec, self.spec_freqs = _make_spectrogram(audio_samples, sample_rate, 200)
        self.frame_count = self.spec.shape[0]
        self.frame_rate = self.frame_count / self.duration
        self.prev_frame_t = 0

        self.canvas = QtGui.QImage(*config.DISPLAY_SHAPE, QtGui.QImage.Format_RGB32)

    def get_frame(self, t):
        dt = t - self.prev_frame_t
        self.prev_frame_t = t

        spec = self.get_spec_frame(t)
        def spec_freq(freq):
            return np.interp(freq, self.spec_freqs, spec)

        # TODO: maybe could do some kind of image processing to automatically identify notes?
        # TODO: capture percussion or at least beat
        # could modulate rotation on beat

        self.canvas.fill(0xFF000000)

        painter = QtGui.QPainter(self.canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(config.DISPLAY_SHAPE[0] / 2, config.DISPLAY_SHAPE[1] / 2)
        # painter.scale(config.DISPLAY_SHAPE[0] / 2, config.DISPLAY_SHAPE[1] / 2)

        def blip(center, strength, hue, saturation, lightness):
            radius = util.lerp(strength, 0, 1, 0.5, 1.75, clip=True)
            brightness = util.lerp(strength, 0, 1, 0.3, 1.0, clip=True)

            grad = QtGui.QRadialGradient(0.5, 0.5, 0.5)
            grad.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
            grad.setColorAt(0.0, QtGui.QColor.fromHsvF(hue, saturation * 0.3, lightness, brightness))
            grad.setColorAt(0.2, QtGui.QColor.fromHsvF(hue, saturation, lightness, brightness * 0.8))
            grad.setColorAt(1.0, QtCore.Qt.transparent)
            painter.setBrush(QtGui.QBrush(grad))
            painter.setPen(QtCore.Qt.NoPen)

            painter.drawEllipse(QtCore.QRectF(
                center[0] - radius,
                center[1] - radius,
                radius * 2,
                radius * 2))

        painter.rotate(util.lerp(t, 0, 20, 0, 360))

        positions = [
            (-2.5, -1.5),
            (1.5, -2.5),
            (2.5, 1.5),
            (-1.5, 2.5),
        ]
        notes = [
            (355.7, 0.65, 0.80),
            (671.0, 0.55, 0.78),
            # (805.1, 0.55, 0.84),
            (1048, 0.45, 0.67),
            (1183, 0.41, 0.59),
        ]
        colors = [
            (146/360, 0.87, 1.00),
            (204/360, 0.91, 1.00),
            (218/360, 0.86, 0.91),
            (231/360, 0.86, 1.00),
        ]

        for position, note, color in zip(positions, notes, colors):
            blip(position, util.lerp(spec_freq(note[0]), note[1], note[2], 0, 1), *color)

        if t >= 22.13:
            blip(
                (0, 0),
                util.lerp(spec_freq(159.7), 0.71, 0.76, 0, 1),
                350/360, 0.85, 1)

        painter.end()

        ptr = self.canvas.constBits()
        ptr.setsize(self.canvas.byteCount())

        pixels = np.array(ptr).reshape(config.DISPLAY_SHAPE[1], config.DISPLAY_SHAPE[0], 4)[:, :, -2:-5:-1].transpose(1, 0, 2)
        return pixels

    def get_spec_frame(self, t):
        return _frame_interp(self.spec, self.frame_rate, t)

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

    dft_size = 1 << 12
    dft = np.fft.rfft(frames, n=dft_size)
    power_spectrum = np.square(np.abs(dft)) / frame_size
    spectrum_freqs = np.fft.rfftfreq(dft_size, d=1 / sample_rate)

    power_spectrum, spectrum_freqs = _mel_filter(power_spectrum, spectrum_freqs, spectrogram_width)
    # power_spectrum = np.log(power_spectrum+1)

    power_spectrum = np.log(power_spectrum + 1)
    power_spectrum /= np.max(power_spectrum)

    power_spectrum = util.gaussian_filter1d(power_spectrum, 2, axis=0)

    return power_spectrum, spectrum_freqs

def _mel_filter(power_spectrum, spectrum_freqs, num_filters):
    def freq_to_mel(f): return 1125 * np.log(1 + f / 700)
    def mel_to_freq(m): return 700 * (np.exp(m / 1125) - 1)

    spec_size = power_spectrum.shape[1]

    min_freq = 20
    # TODO: max freq should be some upper limit, or Nyquist freq?
    max_freq = 4000#self.sample_rate // 2

    min_freq = freq_to_mel(min_freq)
    max_freq = freq_to_mel(max_freq)

    filter_freqs = np.linspace(min_freq, max_freq, num_filters + 2)
    filter_freqs = mel_to_freq(filter_freqs)
    print('spectrum frequencies:')
    print(spectrum_freqs)
    print('filter frequencies:')
    print(filter_freqs)
    filter_freq_idxs = np.interp(filter_freqs, spectrum_freqs, np.arange(spec_size))
    print('filter frequency indicies:')
    print(filter_freq_idxs)

    filterbanks = np.zeros((spec_size, num_filters), dtype=power_spectrum.dtype)
    filterbank_plot = gui.debug_layout.addPlot(
        row=0,
        col=0,
        title='Mel Filterbanks',
        labels={'left': 'Coefficient', 'bottom': 'Frequency'}
    )
    # import matplotlib.pyplot as plt
    for i in range(num_filters):
        filter_min = filter_freq_idxs[i]
        filter_mid = filter_freq_idxs[i + 1]
        filter_max = filter_freq_idxs[i + 2]
        filterbanks[:, i] += np.interp(np.arange(spec_size), np.linspace(filter_min, filter_mid), np.linspace(0, 1, endpoint=False), left=0, right=0)
        filterbanks[:, i] += np.interp(np.arange(spec_size), np.linspace(filter_mid, filter_max), np.linspace(1, 0), left=0, right=0)
        # filterbanks[filter_min : filter_mid, i] = np.linspace(0, 1, filter_mid - filter_min)
        # filterbanks[filter_mid - 1 : filter_max, i] = np.linspace(1, 0, filter_max - filter_mid + 1)
        # filterbanks[:, i] /= ((filter_max - filter_min) / 2)
        # if i < 5:
        #     print(filter_min, filter_mid, filter_max)
        #     print(filterbanks[:, i])
        filterbank_plot.plot(x=spectrum_freqs, y=filterbanks[:, i])
        # plt.plot(filterbanks[:, i])
    # plt.show()

    # print(power_spectrum.shape)
    # print(filterbanks.shape)

    power_spectrum_filtered = np.dot(power_spectrum, filterbanks)

    # print(power_spectrum_filtered.shape)

    return power_spectrum_filtered, filter_freqs[1:-1]
