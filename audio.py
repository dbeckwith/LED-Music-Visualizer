import time
import pydub
import pyaudio
import numpy as np
# from scipy import signal as sig

import config


class Audio(object):
    def __init__(self, path, frame_size, audio_volume=1.0):
        self.path = path
        self.audio_volume = audio_volume

        if self.path == ':debug:':
            print('Generating debug audio')
            self.sample_rate = 44100
            self.channels = 1
            self.sample_width = 2
            self.raw_samples = np.arange(0.0, 10.0, 1 / self.sample_rate)
            self.raw_samples = np.cos(2 * np.pi * (1000 * self.raw_samples + 100 * np.sin(2 * np.pi * 0.5 * self.raw_samples)))
            self.raw_samples = (self.raw_samples * ((1 << 15) - 1)).astype(np.int16)
            print('Generated debug audio')
        else:
            print('Loading audio from {}'.format(path))
            seg = pydub.AudioSegment.from_mp3(path)
            self.sample_rate = seg.frame_rate
            self.channels = seg.channels
            self.sample_width = seg.sample_width
            self.raw_samples = np.array(seg.get_array_of_samples())
            print('Loaded audio')

        self.raw_samples = np.reshape(self.raw_samples, (-1, self.channels))
        self.frame_size = int(frame_size * self.sample_rate)

        self.samples = self.raw_samples.copy()
        self.samples = np.mean(self.samples, axis=1)
        sample_max = (1 << (self.sample_width * 8 - 1)) - 1
        sample_min = ~sample_max
        self.samples = (self.samples - sample_min) / (sample_max - sample_min)
        # self.samples = self.samples / (sample_max + 1)

        print('Creating audio stream')
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
        print('Created audio stream')

        self.running = False

        # self._spectrogram = sig.spectrogram()

        # from matplotlib import pyplot as plt
        # f, t, Sxx = sig.spectrogram(self.samples, self.sample_rate, noverlap=0, mode='magnitude')
        # print(f.shape)
        # print(t.shape)
        # print(Sxx.shape)
        # # plt.plot(f, Sxx[:, 0])
        # plt.pcolormesh(t, f, Sxx)
        # plt.show()
        # exit()

    def _samples(self, t):
        window = np.hamming(self.frame_size)
        sample = int(t * self.sample_rate)
        min_sample = sample - int(np.floor(self.frame_size / 2))
        max_sample = sample + int(np.ceil(self.frame_size / 2))
        if min_sample > self.samples.shape[0] or max_sample < 0:
            samples = np.tile(0.5, (max_sample - min_sample,))
        else:
            min_extra = 0
            if min_sample < 0:
                min_extra = 0 - min_sample
                min_sample = 0
            max_extra = 0
            if max_sample > self.samples.shape[0]:
                max_extra = max_sample - self.samples.shape[0]
                max_sample = self.samples.shape[0]
            samples = np.concatenate((np.tile(0.5, (min_extra,)), self.samples[min_sample:max_sample], np.tile(0.5, (max_extra,))))
        shape = samples.shape
        samples *= window
        return samples

    def spectrogram(self, t):
        # TODO: things to fix
        # overlap window
        # get frame size right
        # scale values using np.fft.fftfreq
        # time filter like scott lawsons' expfilter
        samples = self._samples(t)
        fft = np.fft.rfft(samples, n=config.NUM_LEDS*2)[:config.NUM_LEDS]
        fft = np.abs(fft) ** 2
        # fft = fft * np.exp(np.linspace(0.0, 1.0, fft.shape[0]) - 1) * 10
        # fft = np.concatenate((fft[::-1], fft))
        return fft
        # fft = sig.spectrogram(samples, )
        # pixels = sig.welch(samples, self.sample_rate, )
        # return pixels

    def volume(self, t):
        samples = self._samples(t)
        vol = np.mean(samples ** 0.2)
        pixels = np.zeros((config.NUM_LEDS,), dtype=np.float32)
        vol_pos = vol * pixels.shape[0]
        pixels[:int(np.floor(vol_pos))] = 1.0
        pixels[int(np.floor(vol_pos))] = np.ceil(vol_pos) - np.floor(vol_pos)
        pixels[int(np.ceil(vol_pos)):] = 0.0
        return pixels

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    def start(self):
        print('Starting audio')
        self.running = True
        self.start_time = time.time()
        self.stream.start_stream()
        print('Started audio')

    def stop(self):
        if self.running:
            print('Stopping audio')
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
        print('Stopping audio stream')
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print('Stopped audio stream')
