# -*- coding: utf-8 -*-

import datetime
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

import config
import util


class GUI(object):
    def __init__(self):
        self.closed = True

    def setup(self):
        util.timer('Initializing GUI')

        app = QtGui.QApplication([])

        view = pg.GraphicsView()
        view.resize(1000, 700)
        view.setWindowTitle('LED Music Visualizer')
        view.closeEvent = self._close_event
        self.on_close = None

        layout = pg.GraphicsLayout()
        view.setCentralItem(layout)

        spec_plot = layout.addPlot(title='Spectrogram')
        spec_plot.hideButtons()
        spec_plot.setMouseEnabled(x=False, y=False)
        spec_plot.setYRange(0.0, 1.0, padding=0)
        spec_plot_plot = spec_plot.plot()

        layout.layout.setRowStretchFactor(0, 1)
        layout.nextRow()

        pixel_viewer_proxy = QtGui.QGraphicsProxyWidget(layout)
        pixel_viewer = PixelViewer()
        pixel_viewer_proxy.setWidget(pixel_viewer)
        layout.addItem(pixel_viewer_proxy)

        layout.layout.setRowStretchFactor(1, 1)
        layout.nextRow()

        labels_layout = pg.GraphicsLayout()
        fps_label = labels_layout.addLabel('FPS: ?')
        time_label = labels_layout.addLabel('Elapsed Time: ?')
        layout.addItem(labels_layout)

        layout.layout.setRowStretchFactor(2, 0)

        debug_view = pg.GraphicsView()
        debug_view.resize(800, 600)
        debug_view.setWindowTitle('Debug')

        debug_layout = pg.GraphicsLayout()
        debug_view.setCentralItem(debug_layout)

        self.app = app
        self.view = view
        self.spec_plot_plot = spec_plot_plot
        self.pixel_viewer = pixel_viewer
        self.fps_label = fps_label
        self.time_label = time_label

        self.debug_view = debug_view
        self.debug_layout = debug_layout

    def _close_event(self, event):
        if self.on_close is not None:
            self.on_close()
        event.accept()

    def update_pixels(self, pixels):
        if not self.closed:
            self.pixel_viewer.set_colors(pixels)

    def update_spec(self, spec, spec_freqs):
        if not self.closed:
            self.spec_plot_plot.setData(x=spec_freqs, y=spec)

    def update_fps(self, fps):
        if not self.closed:
            self.fps_label.setText('FPS: {:.1f}'.format(fps))

    def update_time(self, time):
        if not self.closed:
            self.time_label.setText('Elapsed Time: {}'.format(datetime.timedelta(seconds=time)))

    def start(self, show_debug_window=False):
        util.timer('Showing GUI')

        if show_debug_window:
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

class PixelViewer(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        scene = QtGui.QGraphicsScene()
        scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))

        size = 10
        spacing = 4

        self.pixels = np.empty(config.DISPLAY_SHAPE, dtype=np.object_)
        for x, y in np.ndindex(self.pixels.shape):
            pixel = scene.addEllipse(x * (size + spacing), y * (size + spacing), size, size)
            pixel.setPen(QtGui.QPen(QtGui.QBrush(QtGui.QColor(127, 127, 127)), 1.0))
            pixel.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
            self.pixels[x, y] = pixel

        self.setScene(scene)

    def set_colors(self, colors):
        expected_shape = self.pixels.shape + (config.CHANNELS_PER_PIXEL,)
        assert colors.shape == expected_shape, 'colors shape ({}) does not match display shape ({})'.format(colors.shape, expected_shape)
        for coord, pixel in np.ndenumerate(self.pixels):
            pixel.setBrush(QtGui.QBrush(QtGui.QColor(*colors[coord])))

gui = GUI()
