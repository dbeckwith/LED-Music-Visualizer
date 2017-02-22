# -*- coding: utf-8 -*-

import time

import pydub
import pyaudio
import numpy as np
# from scipy import signal as sig

import config
import util
from gui import gui


class Audio(object):
    def __init__(self, path, audio_volume):
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
        self.duration = self.sample_count / self.sample_rate
        print('Samples: {:,d}'.format(self.sample_count))

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

    @property
    def elapsed_time(self):
        return util.lerp(self.stream_pos, 0, self.sample_count, 0, self.duration)

    def start(self):
        util.timer('Starting audio')
        self.running = True
        # self.stream.start_stream()

    def stop(self):
        if self.running:
            util.timer('Stopping audio')
            self.running = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

    def pause(self):
        if self.running:
            self.stream.stop_stream()

    def unpause(self):
        if self.running:
            self.stream.start_stream()

    @property
    def is_paused(self):
        return self.running and self.stream.is_stopped()

    def skip_to(self, time):
        self.stream_pos = int(util.lerp(time, 0, self.duration, 0, self.sample_count))

    def _update_stream(self, in_data, frame_count, time_info, status_flags):
        end = self.stream_pos + frame_count
        if end >= self.samples.shape[0]:
            end = self.samples.shape[0]
            # TODO: should pad with zeros so actually return frame_count frames?
            self.stop()
        data = self.samples[self.stream_pos : end, :].flatten()
        data = (data * self.audio_volume).astype(data.dtype)
        self.stream_pos += frame_count
        return (data, pyaudio.paContinue)
