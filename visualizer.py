import time
import re
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import config


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
    print('Connected')
    return com

class Visualizer(object):
    def __init__(self, use_arduino=True, brightness=1.0):
        self.use_arduino = use_arduino
        self.brightness = np.clip(brightness, 0.0, 1.0)
        self.running = False

    def send_off(self):
        if self.running:
            self._send_off()

    def _send_off(self):
        self._send_pixels(np.zeros((config.NUM_LEDS, 3), dtype=np.uint8))

    def send_pixels(self, pixels):
        if self.running:
            assert pixels.shape == (config.NUM_LEDS, config.NUM_LED_CHANNELS)
            pixels = (np.clip(pixels * self.brightness, 0.0, 1.0) * 0xFF).astype(np.uint8)

            self._send_pixels(pixels)

            for c in range(3):
                self.plots[c].setData(y=pixels[:, c])
            dt = time.time() - self.start_time
            if dt > 0:
                self.fps_label.setText('FPS: {:.1f}'.format(self.frames / dt))

            self.app.processEvents()

            self.frames += 1

            # TODO: if fps too high, graph won't update
            if not self.use_arduino:
                time.sleep(0.01)

    def _send_pixels(self, pixels):
        if self.use_arduino:
            data = []
            for index, color in enumerate(pixels):
                data.append(index)
                data.extend(color)
            self.arduino.write(bytes(data))
            self.arduino.read()

    def start(self):
        print('Starting visualizer')
        self.view.show()
        self.app.processEvents()
        self.start_time = time.time()
        self.frames = 0
        self.running = True
        print('Started visualizer')

    def stop(self):
        if self.running:
            print('Stopping visualizer')
            self.running = False

    def __enter__(self, *args):
        if self.use_arduino:
            self.arduino = _open_arduino_com().__enter__(*args)

        print('Creating GUI')
        # TODO: led image

        self.app = QtGui.QApplication([])

        self.view = pg.GraphicsView()
        self.view.closeEvent = lambda *args: self.stop()
        self.view.resize(800, 600)
        self.view.setWindowTitle('LED Music Visualizer')

        self.layout = pg.GraphicsLayout()
        self.view.setCentralItem(self.layout)

        self.plot = self.layout.addPlot(title='LED Colors')
        self.plot.setYRange(0, 255, padding=0)
        self.plot.getAxis('left').setTickSpacing(15, 3)
        self.plots = []
        for c in range(3):
            self.plots.append(self.plot.plot(pen=tuple(255 if i == c else 0 for i in range(3))))

        self.layout.nextRow()

        self.fps_label = self.layout.addLabel('')

        print('Created GUI')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_arduino:
            print('Disconnecting from Arduino')
            self._send_off()
            self.arduino.close()
            print('Disconnected from Arduino')

        print('Closing GUI')
        self.view.close()
        self.app.quit()
        print('Closed GUI')
