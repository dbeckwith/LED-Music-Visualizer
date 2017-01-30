import pydub
import pyaudio
import numpy as np

import config


class Audio(object):
    def __init__(self, path, frame_size, audio_volume=1.0):
        self.path = path
        self.audio_volume = audio_volume

        seg = pydub.AudioSegment.from_mp3(path)

        self.sample_rate = seg.frame_rate
        self.frame_size = int(frame_size * self.sample_rate)
        self.pad_widths = (int(np.ceil(self.frame_size / 2)), int(np.floor(self.frame_size / 2)))

        self.raw_samples = np.array(seg.get_array_of_samples())
        self.raw_samples = np.reshape(self.raw_samples, (-1, seg.channels))

        self.samples = np.mean(self.raw_samples, axis=1)
        self.samples = np.pad(self.samples, self.pad_widths, mode='constant')
        sample_max = (1 << (seg.sample_width * 8 - 1)) - 1
        sample_min = ~sample_max
        self.samples = (self.samples - sample_min) / (sample_max - sample_min)
        # self.samples = self.samples / (sample_max + 1)

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.sample_rate,
            channels=seg.channels,
            format=pyaudio.get_format_from_width(seg.sample_width, unsigned=False),
            output=True,
            stream_callback=self._update_stream,
            start=False
        )
        self.stream_pos = 0

    def _samples(self, t):
        window = np.hamming(self.frame_size)
        sample = int(t * self.sample_rate)
        min_sample = sample + self.pad_widths[0] - self.pad_widths[0]
        max_sample = sample + self.pad_widths[0] + self.pad_widths[1]
        samples = self.samples[min_sample:max_sample]
        samples *= window
        return samples

    def spectrogram(self, t):
        samples = self._samples(t)
        fft = np.fft.rfft(samples, n=config.NUM_LEDS)[:config.NUM_LEDS // 2]
        fft = np.abs(fft)
        fft = fft ** (1 / 2) * 2
        fft = np.concatenate((fft[::-1], fft))
        return fft

    def volume(self, t):
        samples = self._samples(t)
        vol = np.mean(samples ** 0.2)
        pixels = np.zeros((config.NUM_LEDS,), dtype=np.float32)
        vol_pos = vol * pixels.shape[0]
        pixels[:int(np.floor(vol_pos))] = 1.0
        pixels[int(np.floor(vol_pos))] = np.ceil(vol_pos) - np.floor(vol_pos)
        pixels[int(np.ceil(vol_pos)):] = 0.0
        return pixels

    def start(self):
        self.stream.start_stream()

    def _update_stream(self, in_data, frame_count, time_info, status_flags):
        data = self.raw_samples[self.stream_pos:self.stream_pos+frame_count, :].flatten()
        data = (data * self.audio_volume).astype(data.dtype)
        self.stream_pos += frame_count
        return (data, pyaudio.paContinue)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
