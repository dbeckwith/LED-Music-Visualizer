# -*- coding: utf-8 -*-

import os

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
    cache_path = os.path.join(os.path.dirname(__file__), 'spec_cache.npy')

    def __init__(self, audio_samples, sample_rate):
        self.sample_rate = sample_rate
        self.sample_count = audio_samples.shape[0]
        self.duration = self.sample_count / self.sample_rate

        if os.path.exists(self.cache_path):
            util.timer('Loading cached spectrogram')
            spec_data = np.load(self.cache_path)
        else:
            util.timer('Creating spectrogram')
            spec_data = _make_spectrogram(audio_samples, self.sample_rate, 1 << 9)
            util.timer('Saving spectrogram to cache')
            np.save(self.cache_path, np.array(spec_data))
        self.spec, self.spec_grad, self.spec_freqs = spec_data
        self.spec_idxs = np.arange(len(self.spec_freqs))

        self.frame_count = self.spec.shape[0]
        self.frame_rate = self.frame_count / self.duration

        self.canvas = QtGui.QImage(*config.DISPLAY_SHAPE, QtGui.QImage.Format_RGB32)

    def get_frame(self, t):
        spec = _frame_interp(self.spec, self.frame_rate, t)
        spec_grad = _frame_interp(self.spec_grad, self.frame_rate, t)

        def spec_power(freq):
            return np.interp(freq, self.spec_freqs, spec)

        def spec_peak(min_freq, max_freq):
            from_idx = int(np.floor(np.interp(min_freq, self.spec_freqs, self.spec_idxs)))
            to_idx = int(np.ceil(np.interp(max_freq, self.spec_freqs, self.spec_idxs)))
            spec_range = spec[from_idx : to_idx]
            peak_idx = np.argmax(spec_range) + from_idx
            peak_val = spec[peak_idx]
            peak_freq = np.interp(peak_idx, self.spec_idxs, self.spec_freqs)
            return np.array([util.lerp(peak_freq, min_freq, max_freq, 0, 1), peak_val])

        # TODO: maybe could do some kind of image processing to automatically identify notes?
        # TODO: capture percussion or at least beat
        # could modulate rotation on beat

        self.canvas.fill(0xFF000000)

        painter = QtGui.QPainter(self.canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(config.DISPLAY_SHAPE[0] / 2, config.DISPLAY_SHAPE[1] / 2)
        # painter.scale(config.DISPLAY_SHAPE[0] / 2, config.DISPLAY_SHAPE[1] / 2)

        def blip(center, strength, hue, saturation, lightness):
            if strength == 0:
                return
            hue %= 1

            radius = util.lerp(strength, 0, 1, 0, 1.75, clip=True)
            brightness = util.lerp(strength, 0, 1, 0, 1.0, clip=True)

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

        def edge_glow(strength, hue, saturation, lightness):
            if strength == 0:
                return
            hue %= 1

            radius = 5.5
            brightness = util.lerp(strength, 0, 1, 0, 1.0, clip=True)

            grad = QtGui.QRadialGradient(0.5, 0.5, 0.5)
            grad.setCoordinateMode(QtGui.QGradient.ObjectBoundingMode)
            grad.setColorAt(0.0, QtCore.Qt.transparent)
            grad.setColorAt(0.6, QtCore.Qt.transparent)
            grad.setColorAt(0.8, QtGui.QColor.fromHsvF(hue, saturation, lightness, brightness * 0.8))
            grad.setColorAt(1.0, QtGui.QColor.fromHsvF(hue, saturation * 0.3, lightness, brightness))
            painter.setBrush(QtGui.QBrush(grad))
            painter.setPen(QtCore.Qt.NoPen)

            painter.drawEllipse(QtCore.QRectF(
                -radius,
                -radius,
                radius * 2,
                radius * 2))

        INTRO = 0.20
        BASS = 22.13
        LYRICS_1 = 41.36
        LYRICS_1_KEYBOARD = 63.22
        CHORUS_1 = 85.23
        LYRICS_2 = 107.23
        LYRICS_2_PERCUS = 118.25
        CHORUS_2 = 129.13
        GUITAR_SOLO = 151.12
        GUITAR_SOLO_KEYBOARD = 167.00
        CHORUS_3 = 185.12
        OUTRO = 207.29
        FADE = 226.27
        END = 265.86

        def fade(*args):
            assert args and len(args) % 4 == 0
            for i in range(0, len(args), 4):
                in_t, in_fade, out_t, out_fade = args[i : i + 4]
                if in_fade == 0:
                    if t < in_t:
                        return 0
                elif in_fade < 0:
                    in_fade = -in_fade
                    if t < in_t - in_fade:
                        return 0
                    if t < in_t:
                        return util.lerp(t, in_t - in_fade, in_t, 0, 1)
                else:
                    if t < in_t:
                        return 0
                    if t < in_t + in_fade:
                        return util.lerp(t, in_t, in_t + in_fade, 0, 1)

                if out_fade == 0:
                    if t < out_t:
                        return 1
                elif out_fade < 0:
                    out_fade = -out_fade
                    if t < out_t - out_fade:
                        return 1
                    if t < out_t:
                        return util.lerp(t, out_t - out_fade, out_t, 1, 0)
                else:
                    if t < out_t:
                        return 1
                    if t < out_t + out_fade:
                        return util.lerp(t, out_t, out_t + out_fade, 1, 0)
            return 0

        beat_power = util.lerp(spec_power(3920), 0.4, 0.5, 0, 1, clip=True) * fade(BASS, 0, LYRICS_1, 0)
        bass_pos, bass_power = spec_peak(20, 174) * fade(BASS, 0, END, 0) # TODO: try to get clearer variation in bass_pos
        lyric_pos, lyric_power = spec_peak(523.6, 774.4) * fade(LYRICS_1, 0.1, GUITAR_SOLO, 0)

        # edge_glow(lyric_pos, 82/360, 0.40, 1)

        painter.scale(1, 1)
        painter.rotate(util.lerp(t, 0, 20, 0, 360))
        painter.translate(0 + util.lerp(beat_power, 0, 1, 0, 1), 0)

        for ang, note, color in zip(
            np.linspace(0, 2 * np.pi, 4, endpoint=False), [
                (355.7, 0.65, 0.80),
                (671.0, 0.55, 0.78),
                # (805.1, 0.55, 0.84),
                (1048, 0.45, 0.67),
                (1183, 0.41, 0.59),
            ], [
                (146/360, 0.87, 1.00),
                (204/360, 0.91, 1.00),
                (218/360, 0.86, 0.91),
                (231/360, 0.86, 1.00),
            ]):
            r = 2.91
            note_power = util.lerp(spec_power(note[0]), note[1], note[2], 0, 1, clip=True) * fade(INTRO, 0, LYRICS_1 - 2.84, 2, CHORUS_1, 0, LYRICS_2, -2)
            blip((r * np.cos(ang), r * np.sin(ang)), note_power, *color)

        blip((0, 0), bass_power, util.lerp(bass_pos, 0, 1, 380, 280)/360, 0.85, 1)

        for ang in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            ang += 2 * np.pi / 8
            r = util.lerp(lyric_pos, 0, 1, 2, 4)
            blip((r * np.cos(ang), r * np.sin(ang)), lyric_power, 82/360, 0.40, 1)

        painter.end()

        ptr = self.canvas.constBits()
        ptr.setsize(self.canvas.byteCount())

        pixels = np.array(ptr).reshape(config.DISPLAY_SHAPE[1], config.DISPLAY_SHAPE[0], 4)[:, :, -2:-5:-1].transpose(1, 0, 2)
        return pixels

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
        print('Padding samples by {:,d}'.format(pad_samples))
        samples = np.concatenate((samples, np.zeros(pad_samples)))

    frames = np.empty((frame_count, frame_size), dtype=samples.dtype)
    print('Creating frames of shape {:,d} x {:,d}'.format(frame_count, frame_size))
    for frame_idx in range(frame_count):
        sample_idx = frame_idx * frame_step
        frames[frame_idx] = samples[sample_idx : sample_idx + frame_size]

    print('Applying Hanning window')
    window = np.hanning(frame_size)
    frames *= window

    dft_size = 1 << 13
    print('Calculating RFFT of size {:,d}'.format(dft_size))
    dft = np.fft.rfft(frames, n=dft_size)
    print('Converting DFT to reals')
    power_spectrum = np.square(np.abs(dft)) / frame_size
    print('Calculating spectrum frequencies')
    spectrum_freqs = np.fft.rfftfreq(dft_size, d=1 / sample_rate)

    print('Applying Mel filter')
    power_spectrum, spectrum_freqs = _mel_filter(power_spectrum, spectrum_freqs, spectrogram_width)
    # power_spectrum = np.log(power_spectrum+1)

    print('Scaling spectrum')
    power_spectrum = np.log(power_spectrum + 1)
    power_spectrum /= np.max(power_spectrum)

    print('Calculating spectrum gradient')
    power_spectrum_grad = np.gradient(power_spectrum, 1, axis=0)

    print('Blurring spectrum')
    power_spectrum = util.gaussian_filter1d(power_spectrum, 2, axis=0)
    power_spectrum_grad = util.gaussian_filter1d(power_spectrum_grad, 2, axis=0)

    return power_spectrum, power_spectrum_grad, spectrum_freqs

def _mel_filter(power_spectrum, spectrum_freqs, num_filters):
    def freq_to_mel(f): return 1125 * np.log(1 + f / 700)
    def mel_to_freq(m): return 700 * (np.exp(m / 1125) - 1)

    spec_size = power_spectrum.shape[1]

    min_freq = 20
    # TODO: max freq should be some upper limit, or Nyquist freq?
    max_freq = 4000#self.sample_rate // 2

    min_freq = freq_to_mel(min_freq)
    max_freq = freq_to_mel(max_freq)

    print('Converting frequencies to Mel scale')
    filter_freqs = np.linspace(min_freq, max_freq, num_filters + 2)
    filter_freqs = mel_to_freq(filter_freqs)
    # print('spectrum frequencies:')
    # print(spectrum_freqs)
    # print('filter frequencies:')
    # print(filter_freqs)
    print('Calculating frequency mapping')
    filter_freq_idxs = np.interp(filter_freqs, spectrum_freqs, np.arange(spec_size))
    # print('filter frequency indicies:')
    # print(filter_freq_idxs)

    filterbanks = np.zeros((spec_size, num_filters), dtype=power_spectrum.dtype)
    # filterbank_plot = gui.debug_layout.addPlot(
    #     row=0,
    #     col=0,
    #     title='Mel Filterbanks',
    #     labels={'left': 'Coefficient', 'bottom': 'Frequency'}
    # )
    print('Creating filterbanks')
    for i in range(num_filters):
        filter_min = filter_freq_idxs[i]
        filter_mid = filter_freq_idxs[i + 1]
        filter_max = filter_freq_idxs[i + 2]
        filterbanks[:, i] += np.interp(np.arange(spec_size), np.linspace(filter_min, filter_mid), np.linspace(0, 1, endpoint=False), left=0, right=0)
        filterbanks[:, i] += np.interp(np.arange(spec_size), np.linspace(filter_mid, filter_max), np.linspace(1, 0), left=0, right=0)
        # filterbanks[:, i] /= (filter_max - filter_min) / 2
        # filterbank_plot.plot(x=spectrum_freqs, y=filterbanks[:, i])

    print('Applying filterbanks to spectrum')
    power_spectrum_filtered = np.dot(power_spectrum, filterbanks)

    return power_spectrum_filtered, filter_freqs[1:-1]
