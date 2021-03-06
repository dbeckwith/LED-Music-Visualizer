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
        self.lyric_pos_filter = ExpFilter(0, 0.1, 0.1)
        self.chorus_bg_pos_filter = ExpFilter(0, 0.1, 0.1)

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

        beat_power = util.lerp(spec_power(3920), 0.4, 0.5, 0, 1, clip=True) * fade(BASS, 0, LYRICS_1, 0) + \
            util.lerp(spec_power(3920), 0.39, 0.67, 0, 1, clip=True) * fade(OUTRO, 0, END, 0)
        bass_pos, bass_power = spec_peak(20, 174) * fade(BASS, 0, END, 0) # TODO: try to get clearer variation in bass_pos
        lyric_pos, lyric_power = spec_peak(523.6, 774.4) * fade(
            LYRICS_1, 0.1, GUITAR_SOLO, -2.4,
            CHORUS_3, -1, OUTRO, -2.4)
        freckle_power = fade(45.5, -0.4, 47.1, -0.3)
        chorus_bg_pos, chorus_bg_power = spec_peak(1672, 2169) * fade(
            CHORUS_1, 0, LYRICS_2, -2.4,
            CHORUS_2, 0, GUITAR_SOLO, -2.4,
            CHORUS_3, -1, OUTRO, -2.4)

        self.lyric_pos_filter.update(lyric_pos)
        self.chorus_bg_pos_filter.update(chorus_bg_pos)


        edge_glow(chorus_bg_power, util.lerp(self.chorus_bg_pos_filter.value, 0, 1, 123, 50)/360, 0.80, 0.60)

        # KEYBOARD
        keyboard_blip_pos = [
            (-4, -4),
            (-4,  4),
            ( 4,  4),
            ( 4, -4),
        ]
        keyboard_blip_colors = [
            (146/360, 0.87, 1.00),
            (204/360, 0.91, 1.00),
            (218/360, 0.86, 0.91),
            (231/360, 0.86, 1.00),
        ]
        keyboard_note_offsets = [
             63.31,
             74.28,
            107.22,
            118.18,
            167.70,
            175.82,
        ]
        keyboard_note_timings = [
            (63.31, 0, 63.95,    0),
            (63.68, 0, 64.28,    0),
            (63.99, 0, 64.68,    0),
            (64.36, 0, 65.93, -0.5),

            (66.06, 0, 66.62,    0),
            (66.40, 0, 66.98,    0),
            (66.83, 0, 67.32,    0),
            (67.10, 0, 68.48, -0.5),

            (68.71, 0, 69.43,    0),
            (69.15, 0, 69.50,    0),
            (69.50, 0, 70.02,    0),
            (69.86, 0, 71.37, -0.5),

            (71.26, 0, 71.48,  0.5),
            (71.38, 0, 72.38, -0.5),
        ]
        keyboard_note_blip_indicies = [
            0, 1, 2, 3,

            0, 1, 2, 3,

            0, 2, 1, 3,

            2, 1,
        ]
        for offset in keyboard_note_offsets:
            for blip_index, timing in zip(keyboard_note_blip_indicies, keyboard_note_timings):
                base_offset = keyboard_note_offsets[0]
                timing = (
                    timing[0] - base_offset + offset,
                    timing[1],
                    timing[2] - base_offset + offset,
                    timing[3]
                )
                pos = keyboard_blip_pos[blip_index]
                color = keyboard_blip_colors[blip_index]
                blip(pos, fade(*timing), *color)

        # FRECKLES
        freckle_spacing = np.linspace(-2.5, 2.5, 3)
        for y in range(3):
            for x in range(3):
                if not (x == 1 and y == 1):
                    blip((freckle_spacing[x], freckle_spacing[y]), freckle_power * 0.5, 200/360, 0.5, 1.0)

        painter.save()
        # painter.scale(util.lerp(t, 47.55, 49.20, 1, -1, clip=True) * util.lerp(t, 50.94, 51.93, 1, -1, clip=True), 1)
        painter.scale(1, 1)
        painter.rotate(util.lerp(t, 0, 20, 0, 360))
        painter.translate(0 + util.lerp(beat_power, 0, 1, 0, 1), 0)

        # BLIPS
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
            note_power = util.lerp(spec_power(note[0]), note[1], note[2], 0, 1, clip=True) * fade(
                INTRO, 0, LYRICS_1 - 2.84, 2)
                #GUITAR_SOLO, 0, CHORUS_3, 0)
            blip((r * np.cos(ang), r * np.sin(ang)), note_power, *color)

        # BASS
        blip((0, 0), bass_power, util.lerp(bass_pos, 0, 1, 380, 280)/360, 0.85, 1)

        # LYRICS
        for ang in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            ang += 2 * np.pi / 8
            # spin on verse starts
            ang -= 2 * np.pi / 2 * fade(52.33, 53.22-52.33, GUITAR_SOLO, 0) ** (1 / 2)
            ang -= 2 * np.pi / 2 * fade(63.00, 64.36-63.00, GUITAR_SOLO, 0) ** (1 / 2)
            ang -= 2 * np.pi / 2 * fade(73.99, 75.33-73.99, GUITAR_SOLO, 0) ** (1 / 2)
            ang -= 2 * np.pi / 2 * fade(107.21, 107.93-107.21, GUITAR_SOLO, 0) ** (1 / 2)
            ang -= 2 * np.pi / 2 * fade(118.24, 118.56-118.24, GUITAR_SOLO, 0) ** (1 / 2)
            r = util.lerp(self.lyric_pos_filter.value, 0, 1, 2, 4)
            blip((r * np.cos(ang), r * np.sin(ang)), lyric_power, 82/360, 0.40, 1)

        painter.restore()
        # GUITAR SOLO
        guitar_solo_note_offsets = [
            151.44,
            159.67,
            167.92,
            176.15,
        ]
        guitar_solo_note_timings = [
            151.44, 151.79, 152.13, 152.47, 152.83,
            153.14, 153.34, 153.49, 153.64,

            154.19, 154.53, 154.87, 155.20, 155.56,
            155.90, 156.07, 156.41,

            156.94, 157.25, 157.64, 157.96, 158.32,
            158.65, 158.86, 159.01, 159.27
        ]
        guitar_solo_note_pos = [
                 0,      0,      0,      0,      0,
                 0,      0,      0,      1,

                 1,      1,      1,      1,      1,
                 1,      0,      2,

                 2,      2,      2,      2,      2,
                 1,      2,      2,      3,
        ]
        guitar_solo_blip_offsets = [
            -2.5,
            -1.5,
             1.5,
             2.5,
        ]
        guitar_solo_blip_colors = [
            (43/360, 0.84, 1.00),
            (37/360, 0.84, 0.91),
            (29/360, 0.79, 1.00),
            (20/360, 0.84, 0.91),
        ]
        for time_offset in guitar_solo_note_offsets:
            time_offset = time_offset - guitar_solo_note_offsets[0]
            for pos, timing in zip(guitar_solo_note_pos, guitar_solo_note_timings):
                timing += time_offset
                offset = guitar_solo_blip_offsets[pos]
                color = guitar_solo_blip_colors[pos]
                pos = (
                    offset,
                    util.lerp(
                        t,
                        timing, timing + 0.2,
                        -4, 4))
                blip(pos, 1.0, *color)

        outro_lyrics_note_offsets = [
            211.80,
            220.02,
        ]
        outro_lyrics_note_timings = [
            211.80, 212.52, 213.17,
        ]
        outro_lyrics_note_pos = [
                 0,      2,      1,
        ]
        outro_lyrics_blip_offsets = [
            -2.5,
            -1.5,
             2.5,
        ]
        outro_lyrics_blip_colors = [
            (43/360, 0.84, 1.00),
            (37/360, 0.84, 0.91),
            (29/360, 0.79, 1.00),
        ]
        for time_offset in outro_lyrics_note_offsets:
            time_offset = time_offset - outro_lyrics_note_offsets[0]
            for pos, timing in zip(outro_lyrics_note_pos, outro_lyrics_note_timings):
                timing += time_offset
                offset = outro_lyrics_blip_offsets[pos]
                color = outro_lyrics_blip_colors[pos]
                pos = (
                    offset,
                    util.lerp(
                        util.lerp(
                            t,
                            timing, timing + 0.6,
                            0, 1,
                            clip=True) ** (1 / 2),
                        0, 1,
                        4, -1.5))
                blip(pos, fade(timing, 0, timing + 0.6, -0.4), *color)

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

    print('Applying frame window')
    window = np.hanning(frame_size)
    # n = np.linspace(0, 1, frame_size)
    # window = 0.355768 - 0.487396 * np.cos(2 * np.pi * n) + 0.144232 * np.cos(4 * np.pi * n) - 0.012604 * np.cos(6 * np.pi * n)
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
