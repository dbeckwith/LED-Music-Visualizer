import time
import pydub
import pyaudio
import numpy as np
import pyqtgraph as pg
# from scipy import signal as sig

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
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
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

class Audio(object):
    def __init__(self, path=None, audio_volume=1.0):
        self.path = path
        self.audio_volume = audio_volume

        if self.path is None:
            util.timer('Generating debug audio')
            self.sample_rate = 44100
            self.channels = 1
            self.sample_width = 2
            self.samples = np.arange(0.0, 10.0, 1 / self.sample_rate)
            self.samples = np.cos(2 * np.pi * (1000 * self.samples + 100 * np.sin(2 * np.pi * 0.5 * self.samples)))
            self.samples = (self.samples * ((1 << 15) - 1)).astype(np.int16)
        else:
            util.timer('Loading audio')
            print('Loading audio from {}'.format(path))
            seg = pydub.AudioSegment.from_mp3(path)
            self.sample_rate = seg.frame_rate
            self.channels = seg.channels
            self.sample_width = seg.sample_width
            self.samples = np.array(seg.get_array_of_samples())

        self.samples = np.reshape(self.samples, (-1, self.channels))
        self.sample_count = self.samples.shape[0]

        self._make_spectrogram()

        util.timer('Creating audio stream')
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.sample_rate,
            channels=self.channels,
            format=pyaudio.get_format_from_width(self.sample_width, unsigned=False),
            output=True,
            stream_callback=self._update_stream,
            start=False
        )
        self.stream_pos = 0

        self.running = False

    def _make_spectrogram(self):
        util.timer('Creating spectrogram')

        frame_size = 256
        overlap = frame_size // 2
        hop_size = frame_size - overlap
        frame_count = int(np.ceil(self.sample_count / hop_size)) + 1

        samples = self.samples.copy()
        # print(samples.shape)
        samples = np.mean(samples, axis=1)
        # print(samples.shape)
        sample_max = (1 << (self.sample_width * 8 - 1)) - 1
        sample_min = ~sample_max
        # self.samples = util.lerp(self.samples, sample_min, sample_max, 0, 1)
        samples = samples / (sample_max + 1)
        samples = np.concatenate((np.zeros(frame_size // 2), samples, np.zeros(frame_size // 2)))
        # print(samples.shape)

        # frames = np.empty((frame_count, frame_size), dtype=samples.dtype)
        # print(frames.shape)
        # for frame, sample_idx in enumerate(range(self.sample_count, hop_size)):
        #     try:
        #         frames[frame] = samples[sample_idx : sample_idx + frame_size]
        #     except:
        #         print(frame, sample_idx, sample_idx + frame_size)
        #         exit()
        # print(frame_count, frame_size, (samples.strides[0] * hop_size, samples.strides[0]))

        # TODO: breaks on test signal, probably some samples lengths cause problems, should custom implement
        frames = np.lib.stride_tricks.as_strided(samples, shape=(frame_count, frame_size), strides=(samples.strides[0] * hop_size, samples.strides[0]), writeable=False).copy()

        window = np.hamming(frame_size)
        frames *= window
        spec = np.abs(np.fft.rfft(frames, n=frame_size * 2 - 1))

        time_bins, freq_bins = spec.shape
        # TODO: may need to adjust freq scale
        freq_scale = np.linspace(0, 1, freq_bins, endpoint=False) ** 20
        # print(freq_scale)
        freq_scale *= freq_bins - 1
        gui.debug_layout.addPlot(
            row=0,
            col=0,
            title='Frequency Scale by Frequency',
            labels={'left': 'New Frequency', 'bottom': 'Old Frequency'},
            y=freq_scale
        )
        freq_scale = np.unique(np.round(freq_scale).astype(np.int_))
        # print(freq_scale)
        new_freq_bins = freq_scale.shape[0]

        scaled_spec = np.zeros((time_bins, new_freq_bins), dtype=spec.dtype)
        for i in range(new_freq_bins):
            if i == new_freq_bins - 1:
                scaled_spec[:, i] = np.mean(spec[:, freq_scale[i]:], axis=1)
            else:
                scaled_spec[:, i] = np.mean(spec[:, freq_scale[i]:freq_scale[i+1]], axis=1)
        spec = scaled_spec

        spec = util.lerp(spec, np.percentile(spec, 5), np.percentile(spec, 95), 0, 1)
        np.clip(spec, 0, 1, out=spec)

        power_scale_min = 0.1
        power_scale_max = 2
        power_scale_power = 3
        power_scale = util.lerp(np.linspace(0, 1, spec.shape[1]) ** power_scale_power, 0, 1, power_scale_min, power_scale_max)
        gui.debug_layout.addPlot(
            row=0,
            col=1,
            title='Power Scale by Frequency',
            labels={'left': 'Power Factor', 'bottom': 'Frequency'},
            y=power_scale
        )
        spec *= power_scale

        def power_map(x):
            return x ** (1 / 2)
        debug_power_vals = np.linspace(0, 1, 1000)
        gui.debug_layout.addPlot(
            row=1,
            col=0,
            title='Power Scale by Power',
            labels={'left': 'Power Factor', 'bottom': 'Power'},
            x=debug_power_vals,
            y=power_map(debug_power_vals)
        )
        spec = power_map(spec)

        # hist_vals, hist_bins = np.histogram(spec)
        # gui.debug_layout.addPlot(
        #     row=2,
        #     col=0,
        #     colspan=2,
        #     title='Spectrogram Histogram',
        #     labels={'left': 'Spectrogram Samples', 'bottom': 'Power'},
        #     x=hist_bins[:-1],
        #     y=hist_vals
        # )

        util.gaussian_filter1d(spec, sigma=1, axis=0, output=spec) # blur time axis
        # util.gaussian_filter1d(spec, sigma=0.2, axis=1, output=spec) # blur freq axis

        gui.debug_layout.addPlot(
            row=1,
            col=1,
            title='Average Power by Frequency',
            labels={'left': 'Power', 'bottom': 'Frequency'},
            y=np.mean(spec, axis=0)
        )
        gui.debug_layout.addPlot(
            row=2,
            col=0,
            colspan=2,
            title='Average Power by Time',
            labels={'left': 'Power', 'bottom': 'Time'},
            x=np.linspace(0, self.sample_count / self.sample_rate, spec.shape[0]),
            y=np.mean(spec, axis=1)
        )
        gui.debug_layout.addViewBox(
            row=3,
            col=0,
            colspan=2
        ).addItem(pg.ImageItem(
            image=spec
        ))

        self._spectrogram = spec
        # from matplotlib import pyplot as plt
        # plt.imshow(self._spectrogram[:1000], cmap='gray', vmin=0.0, vmax=1.0)
        # plt.show()

    @smooth(alpha_decay=0.2, alpha_rise=0.99)
    def spectrogram(self, t):
        spec_idx = t * self._spectrogram.shape[0] * self.sample_rate / self.sample_count
        if spec_idx <= 0:
            return self._spectrogram[0]
        if spec_idx >= self._spectrogram.shape[0] - 1:
            return self._spectrogram[-1]
        i1 = np.floor(spec_idx)
        i2 = np.ceil(spec_idx)
        if i1 == i2:
            return self._spectrogram[int(i1)]
        else:
            blend = spec_idx - i1
            return self._spectrogram[int(i1)] * (1 - blend) + self._spectrogram[int(i2)] * blend

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    def start(self):
        util.timer('Starting audio')
        self.running = True
        self.start_time = time.time()
        self.stream.start_stream()

    def stop(self):
        if self.running:
            util.timer('Stopping audio')
            self.running = False

    def _update_stream(self, in_data, frame_count, time_info, status_flags):
        end = self.stream_pos+frame_count
        if end >= self.samples.shape[0]:
            end = self.samples.shape[0]
            self.stop()
        data = self.samples[self.stream_pos:end, :].flatten()
        data = (data * self.audio_volume).astype(data.dtype)
        self.stream_pos += frame_count
        return (data, pyaudio.paContinue)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        util.timer('Stopping audio stream')
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
