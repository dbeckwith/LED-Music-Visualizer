# -*- coding: utf-8 -*-

import socket
import struct
import json

import numpy as np

import config
import util


class Display(object):
    def __init__(self, brightness=1.0):
        self.brightness = np.clip(brightness, 0.0, 1.0)
        self.running = False

    def send_off(self):
        if self.running:
            self._send_off()

    def _send_off(self):
        self._send_pixels(np.zeros(config.DISPLAY_SHAPE + (config.CHANNELS_PER_PIXEL,), dtype=np.uint8))

    def send_pixels(self, pixels):
        if self.running:
            assert pixels.shape == config.DISPLAY_SHAPE + (config.CHANNELS_PER_PIXEL,)
            assert pixels.dtype == np.uint8

            self._send_pixels(pixels)

    def _send_pixels(self, pixels):
        self.fadecandy.send_pixels(pixels)

    def start(self):
        util.timer('Connecting to LEDs')
        self.running = True
        self.fadecandy = FadeCandy()
        self.fadecandy.connect()
        self.fadecandy.send_color_settings(
            gamma=2.5,
            r=self.brightness * 1.0,
            g=self.brightness * 0.8,
            b=self.brightness * 0.7
        )

    def stop(self):
        if self.running:
            util.timer('Disconnecting from LEDs')
            self.running = False
            self._send_off()
            self.fadecandy.disconnect()

# http://openpixelcontrol.org/
class FadeCandy(object):
    MAX_LEDS_PER_CHANNEL = 64

    def __init__(self):
        self.host = config.FADECANDY_HOST
        self.port = config.FADECANDY_PORT

        self.socket = None

    def connect(self):
        if self.socket is not None:
            self.disconnect()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # TODO: better error messages
        self.socket.connect((self.host, self.port))
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

    def send_color_settings(self, gamma, r, g, b):
        msg = bytes(json.dumps({'gamma': gamma, 'whitepoint': [r, g, b]}), 'utf-8')
        header = struct.pack('>BBHHH', 0, 0xFF, 4 + len(msg), 1, 1)
        self.socket.send(header + msg)

    def send_pixels(self, pixels):
        assert pixels.ndim in (2, 3)
        assert pixels.shape[-1] == 3
        assert pixels.dtype == np.uint8
        if pixels.ndim == 2:
            for i in range(pixels.shape[0] // FadeCandy.MAX_LEDS_PER_CHANNEL):
                b = bytes(pixels[i : min(i + FadeCandy.MAX_LEDS_PER_CHANNEL, pixels.shape[0]), :])
                header = struct.pack('>BBH', i + 1, 0x00, len(b))
                self.socket.send(header + b)
        else:
            channel = 0
            dimmax = int(np.sqrt(FadeCandy.MAX_LEDS_PER_CHANNEL))
            for y in range(pixels.shape[1] // dimmax):
                for x in range(pixels.shape[0] // dimmax):
                    b = bytes(pixels[x : min(x + dimmax, pixels.shape[0]), y : min(y + dimmax, pixels.shape[1]), :].transpose(1, 0, 2))
                    header = struct.pack('>BBH', channel, 0x00, len(b))
                    self.socket.send(header + b)
                    channel += 1

    def disconnect(self):
        if self.socket is not None:
            self.socket.close()
