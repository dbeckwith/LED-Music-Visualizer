import time
import serial
from serial.tools.list_ports import comports as list_ports
import matplotlib.pyplot as plt

import config


def _get_arduino_port():
    for info in list_ports():
        # http://www.linux-usb.org/usb.ids
        if info.vid == 0x2341:
            return info.device
    return None

def _open_arduino_com():
    port = _get_arduino_port()
    if port is None:
        print('No Arduino connected')
        exit()
    print('Connecting to Arduino on {}...'.format(port))
    com = serial.Serial(
        port = port,
        baudrate = 115200,
        timeout = 1.0
    )
    time.sleep(3)
    print('Connected')
    return com

class Visualizer(object):
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def send_pixels(self, pixels):
        if not self.debug_mode:
            data = []
            for index, color in enumerate(pixels):
                data.append(index)
                for c in color:
                    if c < 0.0: c = 0.0
                    if c > 1.0: c = 1.0
                    data.append(int(c * 0xFF))
            self.arduino.write(bytes(data))
            self.arduino.read()
        else:
            # TODO: change to opengl window or something
            plot_args = []
            for i, c in enumerate('rgb'):
                plot_args.append(pixels[:, i])
                plot_args.append(c + '-')
            plt.cla()
            plt.plot(*plot_args)
            plt.pause(0.001)

    def __enter__(self, *args):
        if not self.debug_mode:
            self.arduino = _open_arduino_com().__enter__(*args)
        else:
            plt.axis([0, config.NUM_LEDS-1, 0.0, 1.0])
            plt.ion()
        return self

    def __exit__(self, *args):
        if not self.debug_mode:
            self.arduino.__exit__(*args)
