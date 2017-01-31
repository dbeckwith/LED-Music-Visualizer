import datetime
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import config
import util


class GUI(object):
    def __init__(self):
        util.timer('Initializing GUI')

        app = QtGui.QApplication([])

        view = pg.GraphicsView()
        view.resize(1000, 700)
        view.setWindowTitle('LED Music Visualizer')

        layout = pg.GraphicsLayout()
        view.setCentralItem(layout)

        spec_plot = layout.addPlot(title='Spectrogram')
        spec_plot.hideButtons()
        spec_plot.setMouseEnabled(x=False, y=False)
        spec_plot.setYRange(0.0, 1.0, padding=0)
        spec_plot_plot = spec_plot.plot()

        layout.layout.setRowStretchFactor(0, 1)
        layout.nextRow()

        # TODO: make into bar graph https://stackoverflow.com/questions/36551044/how-to-plot-two-barh-in-one-axis-in-pyqtgraph
        colors_plot = layout.addPlot(title='LED Colors')
        colors_plot.hideButtons()
        colors_plot.setMouseEnabled(x=False, y=False)
        colors_plot.setYRange(0.0, 1.0, padding=0)
        channel_plots = []
        for c in range(3):
            channel_plots.append(colors_plot.plot(pen=tuple(255 if i == c else 0 for i in range(3))))

        layout.layout.setRowStretchFactor(1, 1)
        layout.nextRow()

        led_viewer_proxy = QtGui.QGraphicsProxyWidget(layout)
        led_viewer = LEDViewer()
        led_viewer_proxy.setWidget(led_viewer)
        layout.addItem(led_viewer_proxy)

        layout.layout.setRowStretchFactor(2, 0)
        layout.nextRow()

        labels_layout = pg.GraphicsLayout()
        fps_label = labels_layout.addLabel('FPS: ?')
        time_label = labels_layout.addLabel('Elapsed Time: ?')
        layout.addItem(labels_layout)

        layout.layout.setRowStretchFactor(3, 0)

        debug_view = pg.GraphicsView()
        debug_view.resize(800, 600)
        debug_view.setWindowTitle('Debug')

        debug_layout = pg.GraphicsLayout()
        debug_view.setCentralItem(debug_layout)

        self.app = app
        self.view = view
        self.spec_plot_plot = spec_plot_plot
        self.channel_plots = channel_plots
        self.led_viewer = led_viewer
        self.fps_label = fps_label
        self.time_label = time_label

        self.debug_view = debug_view
        self.debug_layout = debug_layout

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

    def update_time(self, time):
        if not self.closed:
            self.time_label.setText('Elapsed Time: {}'.format(datetime.timedelta(seconds=time)))

    def add_debug_plot(self, *args, **kwargs):
        return self.debug_layout.addPlot(*args, **kwargs)

    def start(self):
        util.timer('Showing GUI')

        self.debug_view.show()
        self.view.show()
        self.app.processEvents()
        self.closed = False

    def stop(self):
        util.timer('Closing GUI')

        self.closed = True
        self.view.close()
        self.debug_view.close()
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
