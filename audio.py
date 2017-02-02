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
    def __init__(self, path, audio_volume, spectrogram_width):
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
        print('Samples: {:,d}'.format(self.sample_count))

        self._spectrogram = self._make_spectrogram(spectrogram_width)

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

    def _make_spectrogram(self, spectrogram_width):
        util.timer('Creating spectrogram')

        frame_size = int(25 / 1000 * self.sample_rate)
        frame_step = frame_size // 2

        samples = self.samples.copy()
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

        window = np.hamming(frame_size)
        frames *= window

        dft_size = 1024
        dft = np.fft.rfft(frames, n=dft_size)
        power_spectrum = np.square(np.abs(dft)) / frame_size
        spectrum_freqs = np.fft.rfftfreq(dft_size, d=1 / self.sample_rate)

        power_spectrum = self._mel_filter(power_spectrum, spectrum_freqs, spectrogram_width)
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

    def _mel_filter(self, power_spectrum, spectrum_freqs, num_filters):
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
