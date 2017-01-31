import time
import re
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

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
    import serial
    port = _get_arduino_port()
    if port is None:
        print('No Arduino connected')
        exit()
    util.timer('Connecting to Arduino'.format(port))
    print('Connecting to Arduino on {}'.format(port))
    com = serial.Serial(
        port = port,
        baudrate = 115200,
        timeout = 1.0
    )
    time.sleep(3)
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
            assert pixels.shape in ((config.NUM_LEDS,), (config.NUM_LEDS, config.NUM_LED_CHANNELS))
            pixels = (np.clip(pixels * self.brightness, 0.0, self.brightness) * 0xFF).astype(np.uint8)
            if pixels.ndim == 1:
                pixels = np.tile(pixels, (config.NUM_LED_CHANNELS, 1)).T

            self._send_pixels(pixels)

            for i, p in enumerate(self.channel_plots):
                p.setData(y=pixels[:, i])

            dt = time.time() - self.start_time
            if dt > 0:
                self.fps_label.setText('FPS: {:.1f}'.format(self.frames / dt))

            self.led_viewer.set_colors((pixels / self.brightness).astype(np.uint8))

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

    def update_spec(self, spec):
        self.spec_plot_plot.setData(y=spec)

    def start(self):
        util.timer('Starting visualizer')
        self.view.show()
        self.app.processEvents()
        self.start_time = time.time()
        self.frames = 0
        self.running = True

    def stop(self):
        if self.running:
            util.timer('Stopping visualizer')
            self.running = False

    def __enter__(self, *args):
        if self.use_arduino:
            self.arduino = _open_arduino_com().__enter__(*args)

        util.timer('Creating GUI')

        self.app = QtGui.QApplication([])

        self.view = pg.GraphicsView()
        self.view.closeEvent = lambda *args: self.stop()
        self.view.resize(1000, 700)
        self.view.setWindowTitle('LED Music Visualizer')

        self.layout = pg.GraphicsLayout()
        self.view.setCentralItem(self.layout)

        self.spec_plot = self.layout.addPlot(title='Spectrogram')
        self.spec_plot.hideButtons()
        self.spec_plot.setMouseEnabled(x=False, y=False)
        self.spec_plot.setYRange(0.0, 1.0, padding=0)
        self.spec_plot_plot = self.spec_plot.plot()
        self.layout.layout.setRowStretchFactor(0, 1)

        self.layout.nextRow()

        # TODO: make into bar graph https://stackoverflow.com/questions/36551044/how-to-plot-two-barh-in-one-axis-in-pyqtgraph
        self.colors_plot = self.layout.addPlot(title='LED Colors')
        self.colors_plot.hideButtons()
        self.colors_plot.setMouseEnabled(x=False, y=False)
        self.colors_plot.setYRange(0, 255, padding=0)
        self.colors_plot.getAxis('left').setTickSpacing(15, 3)
        self.channel_plots = []
        for c in range(3):
            self.channel_plots.append(self.colors_plot.plot(pen=tuple(255 if i == c else 0 for i in range(3))))
        self.layout.layout.setRowStretchFactor(1, 1)

        self.layout.nextRow()

        led_viewer_proxy = QtGui.QGraphicsProxyWidget(self.layout)
        self.led_viewer = LEDViewer()
        led_viewer_proxy.setWidget(self.led_viewer)
        self.layout.addItem(led_viewer_proxy)
        self.layout.layout.setRowStretchFactor(2, 0)

        self.layout.nextRow()

        self.fps_label = self.layout.addLabel('')
        self.layout.layout.setRowStretchFactor(3, 0)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_arduino:
            util.timer('Disconnecting from Arduino')
            self._send_off()
            self.arduino.close()

        util.timer('Closing GUI')
        self.view.close()
        self.app.quit()
        # pg.exit()

class LEDViewer(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        scene = QtGui.QGraphicsScene()
        scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))

        size = 10
        spacing = 4

        self.leds = []
        for i in range(config.NUM_LEDS):
            led = scene.addEllipse(i * (size + spacing), 0.0, size, size)
            led.setPen(QtGui.QPen(QtGui.QBrush(QtGui.QColor(127, 127, 127)), 1.0))
            led.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
            self.leds.append(led)

        self.setScene(scene)

    def set_colors(self, colors):
        for i, led in enumerate(self.leds):
            led.setBrush(QtGui.QBrush(QtGui.QColor(*colors[i])))
