import time
import re
import numpy as np

import config
import util


def _get_arduino_port():
    from serial.tools.list_ports import comports as list_ports
    for info in list_ports():
        vendor = getattr(info, 'vid', None)
        if vendor is None:
            try:
                m = re.search(r'VID(?::PID)?=(?P<vendor>[0-9a-fA-F]+)', info[2])
                vendor = int(m.group('vendor'), base=16)
            except:
                vendor = None
        # http://www.linux-usb.org/usb.ids
        if vendor == 0x2341:
            port = getattr(info, 'device', None)
            if port is None:
                port = info[0]
            return port
    return None

def _open_arduino_com():
    util.timer('Connecting to Arduino')
    import serial
    port = _get_arduino_port()
    if port is None:
        print('No Arduino connected')
        exit()
    print('Connecting to Arduino on {}'.format(port))
    com = serial.Serial(
        port = port,
        baudrate = 115200,
        timeout = 1.0
    )
    time.sleep(3)
    return com

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
            pixels = (np.clip(pixels * self.brightness, 0.0, self.brightness) * 0xFF).astype(np.uint8)
            if pixels.ndim == 1:
                pixels = np.tile(pixels, (config.NUM_LED_CHANNELS, 1)).T

            self._send_pixels(pixels)

    def _send_pixels(self, pixels):
        if self.use_leds:
            data = []
            for index, color in enumerate(pixels):
                data.append(index)
                data.extend(color)
            self.arduino.write(bytes(data))
            self.arduino.read()

    def start(self):
        util.timer('Starting visualizer')
        self.running = True

    def stop(self):
        if self.running:
            util.timer('Stopping visualizer')
            self.running = False

    def __enter__(self, *args):
        if self.use_leds:
            self.arduino = _open_arduino_com().__enter__(*args)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_leds:
            util.timer('Disconnecting from Arduino')
            self._send_off()
            self.arduino.close()
