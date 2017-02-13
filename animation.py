# -*- coding: utf-8 -*-

import os.path

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

        # TODO: need to make spectrogram much more detailed, also maybe no Mel filter
        self.spec, self.spec_freqs = _make_spectrogram(audio_samples, sample_rate, 200)
        self.frame_count = self.spec.shape[0]
        self.frame_rate = self.frame_count / self.duration
        self.prev_frame_t = 0
        self.spec_frame_t = None

        self.canvas = QtGui.QImage(*config.DISPLAY_SHAPE, QtGui.QImage.Format_RGB32)

    def get_frame(self, t):
        dt = t - self.prev_frame_t
        self.prev_frame_t = t

        spec = self.get_spec_frame(t)
        def spec_freq(freq):
            return np.interp(freq, self.spec_freqs, spec)

        # TODO: maybe could do some kind of peak analysis to get specific instruments?

        self.canvas.fill(0xFF000000)

        painter = QtGui.QPainter(self.canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(config.DISPLAY_SHAPE[0] / 2, config.DISPLAY_SHAPE[1] / 2)
        # painter.scale(config.DISPLAY_SHAPE[0] / 2, config.DISPLAY_SHAPE[1] / 2)

        def blip(center, strength, hue):
            radius = util.lerp(strength, 0, 1, 0.5, 1.75, clip=True)
            brightness = util.lerp(strength, 0, 1, 0.3, 1.0, clip=True)

            grad = QtGui.QRadialGradient(0.5, 0.5, 0.5)
            grad.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
            grad.setColorAt(0.0, QtGui.QColor.fromHsvF(hue, 0.40, 1, brightness))
            grad.setColorAt(0.2, QtGui.QColor.fromHsvF(hue, 0.82, 1, brightness * 0.8))
            grad.setColorAt(1.0, QtCore.Qt.transparent)
            painter.setBrush(QtGui.QBrush(grad))
            painter.setPen(QtCore.Qt.NoPen)

            painter.drawEllipse(QtCore.QRectF(
                center[0] - radius,
                center[1] - radius,
                radius * 2,
                radius * 2))

        painter.rotate(util.lerp(t, 0, 20, 0, 360))

        blip(
            (-2.5, -1.5),
            util.lerp(spec_freq(650), 0.6, 0.75, 0, 1),
            232/360)
        blip(
            (1.5, -2.5),
            util.lerp(spec_freq(1050), 0.5, 0.6, 0, 1),
            232/360)
        blip(
            (2.5, 1.5),
            util.lerp(spec_freq(1175), 0.4, 0.5, 0, 1),
            232/360)
        blip(
            (-1.5, 2.5),
            util.lerp(spec_freq(1550), 0.45, 0.55, 0, 1),
            232/360)

        if t >= 22:
            blip(
                (0, 0),
                util.lerp(spec_freq(300), 0.6, 0.8, 0, 1),
                350/360)

        painter.end()

        ptr = self.canvas.constBits()
        ptr.setsize(self.canvas.byteCount())

        pixels = np.array(ptr).reshape(config.DISPLAY_SHAPE[1], config.DISPLAY_SHAPE[0], 4)[:, :, -2:-5:-1].transpose(1, 0, 2)
        return pixels

    @smooth(alpha_decay=0.2, alpha_rise=0.99, key='anispec')
    def get_spec_frame(self, t):
        if self.spec_frame_t == None or self.spec_frame_t != t:
            self.spec_frame_t = t
            return _frame_interp(self.spec, self.frame_rate, t)
        else:
            return _smoothing_filters['anispec'].value

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

    print(np.min(power_spectrum), np.max(power_spectrum))
    power_spectrum = np.log(power_spectrum + 1)
    power_spectrum /= np.max(power_spectrum)

    # import matplotlib.pyplot as plt
    # plt.imshow(power_spectrum[10000:11000].T, cmap='gray')
    # plt.show()

    # exit()

    img_plot = gui.debug_layout.addPlot(
        row=1,
        col=0
    )

    img_item = pg.ImageItem(
        image=power_spectrum
    )
    img_plot.addItem(img_item)
    # img_plot.addLine
    # img_plot.getAxis('left').tickValues(spectrum_freqs[0], spectrum_freqs[-1], spectrum_freqs.shape[0])
    # img_plot.getAxis('bottom').tickValues(0, self.duration, 1000)

    spec_img = util.lerp(power_spectrum, 0, 1, 0, 0xFF, clip=True).astype(np.uint8)
    spec_img = np.repeat(spec_img[:, :, np.newaxis], 4, 2)
    spec_img = spec_img.transpose(1, 0, 2)
    QtGui.QImage(spec_img.tobytes(), spec_img.shape[1], spec_img.shape[0], QtGui.QImage.Format_RGB32).save(os.path.join(os.path.dirname(__file__), 'spec.png'))

    return power_spectrum, spectrum_freqs

def _mel_filter(power_spectrum, spectrum_freqs, num_filters):
    def freq_to_mel(f): return 1125 * np.log(1 + f / 700)
    def mel_to_freq(m): return 700 * (np.exp(m / 1125) - 1)

    print(power_spectrum.shape)

    spec_size = power_spectrum.shape[1]

    min_freq = 20
    # TODO: max freq should be some upper limit, or Nyquist freq?
    max_freq = 4000#self.sample_rate // 2

    min_freq = freq_to_mel(min_freq)
    max_freq = freq_to_mel(max_freq)

    filter_freqs = np.linspace(min_freq, max_freq, num_filters + 2)
    filter_freqs = mel_to_freq(filter_freqs)
    print(filter_freqs)
    print(spectrum_freqs)
    # TODO: this rebinning with the filters could be more accurate?
    # this round really impacts lower frequency bins
    filter_freq_idxs = np.interp(filter_freqs, spectrum_freqs, np.arange(spec_size))
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
