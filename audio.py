import time
import pydub
import pyaudio
import numpy as np
# from scipy import signal as sig

import config
import util


class Audio(object):
    def __init__(self, path, audio_volume=1.0):
        self.path = path
        self.audio_volume = audio_volume

        if self.path == ':debug:':
            util.timer('Generating debug audio')
            self.sample_rate = 44100
            self.channels = 1
            self.sample_width = 2
            self.raw_samples = np.arange(0.0, 10.0, 1 / self.sample_rate)
            self.raw_samples = np.cos(2 * np.pi * (1000 * self.raw_samples + 100 * np.sin(2 * np.pi * 0.5 * self.raw_samples)))
            self.raw_samples = (self.raw_samples * ((1 << 15) - 1)).astype(np.int16)
        else:
            util.timer('Loading audio from {}'.format(path))
            print('Loading audio from {}'.format(path))
            seg = pydub.AudioSegment.from_mp3(path)
            self.sample_rate = seg.frame_rate
            self.channels = seg.channels
            self.sample_width = seg.sample_width
            self.raw_samples = np.array(seg.get_array_of_samples())

        self.raw_samples = np.reshape(self.raw_samples, (-1, self.channels))
        self.sample_count = self.raw_samples.shape[0]

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

        # TODO: things to fix
        # overlap window
        # get frame size right
        # scale values using np.fft.fftfreq
        # time filter like scott lawsons' expfilter

        overlap = 0.5
        frame_size = 128

        samples = self.raw_samples.copy()
        samples = np.mean(samples, axis=1)
        sample_max = (1 << (self.sample_width * 8 - 1)) - 1
        sample_min = ~sample_max
        # self.samples = (self.samples - sample_min) / (sample_max - sample_min)
        samples = samples / (sample_max + 1)
        samples = np.concatenate((np.zeros(int(np.floor(frame_size / 2))), samples, np.zeros(int(np.ceil(frame_size / 2)))))

        hop_size = frame_size - int(np.floor(overlap * frame_size))
        frame_count = int(np.ceil(self.sample_count / hop_size)) + 1
        # print(frame_size, frame_count)
        frames = np.lib.stride_tricks.as_strided(samples, shape=(frame_count, frame_size), strides=(samples.strides[0] * hop_size, samples.strides[0]))
        window = np.hamming(frame_size)
        frames *= window
        spec = np.fft.rfft(frames, n=frame_size * 2 - 1)
        # print(spec.shape)
        # TODO: resulting spec too small? too much interpolation neccessary

        time_bins, freq_bins = spec.shape
        freq_scale = np.linspace(0, 1, freq_bins) ** 1
        freq_scale *= freq_bins - 1
        freq_scale = np.unique(np.round(freq_scale).astype(np.int_))
        new_freq_bins = freq_scale.shape[0]

        scaled_spec = np.zeros((time_bins, new_freq_bins), dtype=np.complex128)
        for i in range(new_freq_bins):
            if i == new_freq_bins - 1:
                scaled_spec[:, i] = np.sum(spec[:, freq_scale[i]:], axis=1)
            else:
                scaled_spec[:, i] = np.sum(spec[:, freq_scale[i]:freq_scale[i+1]], axis=1)

        self._spectrogram = scaled_spec
        # from matplotlib import pyplot as plt
        # plt.imshow(np.abs(self._spectrogram))
        # plt.show()
        # exit()

    def spectrogram(self, t):
        spec_idx = t * self._spectrogram.shape[0] * self.sample_rate / self.sample_count
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
        if end >= self.raw_samples.shape[0]:
            end = self.raw_samples.shape[0]
            self.stop()
        data = self.raw_samples[self.stream_pos:end, :].flatten()
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
