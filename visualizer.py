import socket
import struct
import json
import numpy as np

import config
import util


class Visualizer(object):
    def __init__(self, use_leds=True, brightness=1.0):
        self.use_leds = use_leds
        self.brightness = np.clip(brightness, 0.0, 1.0)
        self.running = False

    def send_off(self):
        if self.running:
            self._send_off()

    def _send_off(self):
        self._send_pixels(np.zeros((config.NUM_LEDS, 3), dtype=np.uint8))

    def send_pixels(self, pixels):
        if self.running:
            assert pixels.shape in ((config.NUM_LEDS,), (config.NUM_LEDS, config.NUM_LED_CHANNELS))
            pixels = (np.clip(pixels, 0.0, 1.0) * 0xFF).astype(np.uint8)
            if pixels.ndim == 1:
                pixels = np.tile(pixels, (config.NUM_LED_CHANNELS, 1)).T

            self._send_pixels(pixels)

    def _send_pixels(self, pixels):
        if self.use_leds:
            self.fadecandy.send_pixels(pixels)

    def start(self):
        util.timer('Starting visualizer')
        self.running = True

    def stop(self):
        if self.running:
            util.timer('Stopping visualizer')
            self.running = False

    def __enter__(self, *args):
        if self.use_leds:
            util.timer('Connecting to LEDs')
            self.fadecandy = FadeCandy()
            self.fadecandy.connect()
            self.fadecandy.send_color_settings(
                gamma=2.5,
                r=self.brightness * 1.0,
                g=self.brightness * 0.8,
                b=self.brightness * 0.7
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_leds:
            util.timer('Disconnecting from LEDs')
            self._send_off()
            self.fadecandy.disconnect()

# http://openpixelcontrol.org/
class FadeCandy(object):
    def __init__(self):
        self.host = config.FADECANDY_HOST
        self.port = config.FADECANDY_PORT

        self.socket = None

    def connect(self):
        if self.socket is not None:
            self.disconnect()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

    def send_color_settings(self, gamma, r, g, b):
        msg = bytes(json.dumps({'gamma': gamma, 'whitepoint': [r, g, b]}), 'utf-8')
        header = struct.pack('>BBHHH', 0, 0xFF, 4 + len(msg), 1, 1)
        self.socket.send(header + msg)

    def send_pixels(self, pixels, channel=0):
        assert pixels.ndim == 2
        assert pixels.shape[1] == 3
        assert pixels.dtype == np.uint8
        pixels = bytes(pixels)
        header = struct.pack('>BBH', channel, 0x00, len(pixels))
        self.socket.send(header + pixels)

    def disconnect(self):
        if self.socket is not None:
            self.socket.close()
