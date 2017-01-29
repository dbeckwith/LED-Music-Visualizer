import time
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import config


def _get_arduino_port():
    from serial.tools.list_ports import comports as list_ports
    for info in list_ports():
        # http://www.linux-usb.org/usb.ids
        if info.vid == 0x2341:
            return info.device
    return None

def _open_arduino_com():
    import serial
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
    def __init__(self, use_arduino=True, brightness=1.0):
        self.use_arduino = use_arduino
        self.brightness = np.clip(brightness, 0.0, 1.0)
        self.stopped = False

    def send_off(self):
        if not self.stopped:
            self._send_off()

    def _send_off(self):
        self._send_pixels(np.zeros((config.NUM_LEDS, 3), dtype=np.float32))

    def send_pixels(self, pixels):
        if not self.stopped:
            self._send_pixels(pixels)

    def _send_pixels(self, pixels):
        pixels = (np.clip(pixels * self.brightness, 0.0, 1.0) * 0xFF).astype(np.uint8)

        if self.use_arduino:
            data = []
            for index, color in enumerate(pixels):
                data.append(index)
                data.extend(color)
            self.arduino.write(bytes(data))
            self.arduino.read()

        for c in range(3):
            self.plots[c].setData(y=pixels[:, c])
        self.app.processEvents()

    def stop(self):
        if not self.stopped:
            print('Stopping vis')
            self.stopped = True
            if self.use_arduino:
                self._send_off()
                self.arduino.close()

            self.win.close()

    def __enter__(self, *args):
        if self.use_arduino:
            self.arduino = _open_arduino_com().__enter__(*args)

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='LED Music Visualizer')
        self.win.closeEvent = lambda *args: self.stop()
        self.win.resize(800, 600)
        self.layout = QtGui.QHBoxLayout()
        self.win.setLayout(self.layout)
        self.plot_widget = pg.PlotWidget(title='LED Colors')
        self.plot_widget.setYRange(0, 255, padding=0)
        self.plot_widget.getAxis('left').setTickSpacing(15, 3)
        self.layout.addWidget(self.plot_widget)
        self.plots = []
        for c in range(3):
            self.plots.append(self.plot_widget.plot(pen=tuple(255 if i == c else 0 for i in range(3))))
        return self

    def __exit__(self, *args):
        self.stop()
