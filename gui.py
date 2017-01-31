import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import config
import util


class GUI(object):
    def __init__(self):
        util.timer('Initializing GUI')

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
        self.colors_plot.setYRange(0.0, 1.0, padding=0)
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

        self.fps_label = self.layout.addLabel('FPS: ?')
        self.layout.layout.setRowStretchFactor(3, 0)

        self.closed = True

    def update_leds(self, pixels):
        if not self.closed:
            for i, p in enumerate(self.channel_plots):
                p.setData(y=pixels[:, i])

            self.led_viewer.set_colors(pixels)

    def update_spec(self, spec):
        if not self.closed:
            self.spec_plot_plot.setData(y=spec)

    def update_fps(self, fps):
        if not self.closed:
            self.fps_label.setText('FPS: {:.1f}'.format(fps))

    def start(self):
        util.timer('Showing GUI')

        self.view.show()
        self.app.processEvents()
        self.closed = False

    def stop(self):
        util.timer('Closing GUI')

        self.closed = True
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
        colors = (colors * 0xFF).astype(np.uint8)
        for i, led in enumerate(self.leds):
            led.setBrush(QtGui.QBrush(QtGui.QColor(*colors[i])))

gui = GUI()
