# -*- coding: utf-8 -*-

import datetime
import os.path

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

import config
import util


class GUI(object):
    def __init__(self):
        self.closed = True
        self.pause_requested = False

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

        spec_viewer_proxy = QtGui.QGraphicsProxyWidget(layout)
        spec_viewer = SpectrogramViewer()
        spec_viewer_proxy.setWidget(spec_viewer)
        layout.addItem(spec_viewer_proxy)

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
        pause_label = labels_layout.addLabel('Pause')
        pause_label.mousePressEvent = self.__pause_pressed
        layout.addItem(labels_layout)

        layout.layout.setRowStretchFactor(2, 0)

        debug_view = pg.GraphicsView()
        debug_view.resize(800, 600)
        debug_view.setWindowTitle('Debug')

        debug_layout = pg.GraphicsLayout()
        debug_view.setCentralItem(debug_layout)

        self.app = app
        self.view = view
        self.spec_viewer = spec_viewer
        self.pixel_viewer = pixel_viewer
        self.fps_label = fps_label
        self.time_label = time_label
        self.pause_label = pause_label

        self.debug_view = debug_view
        self.debug_layout = debug_layout

    def __pause_pressed(self, event):
        self.pause_requested = True
        if self.pause_label.text == 'Pause':
            self.pause_label.setText('Resume')
        else:
            self.pause_label.setText('Pause')

    def _close_event(self, event):
        if self.on_close is not None:
            self.on_close()
        event.accept()

    def update_pixels(self, pixels):
        if not self.closed:
            self.pixel_viewer.set_colors(pixels)

    def update_fps(self, fps):
        if not self.closed:
            self.fps_label.setText('FPS: {:.1f}'.format(fps))

    def update_time(self, time):
        if not self.closed:
            self.time_label.setText('Elapsed Time: {}'.format(datetime.timedelta(seconds=time)))
            self.spec_viewer.update_time(time)

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

class SpectrogramViewer(QtGui.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(self.layout)

        self.view = QtGui.QGraphicsView(self)
        self.layout.addWidget(self.view)

        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.view.setMouseTracking(True)

        self.scene = QtGui.QGraphicsScene(self)
        self.scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        self.view.setScene(self.scene)

        self.value_label = QtGui.QLabel(self)
        self.layout.addWidget(self.value_label)

    def set_spectrogram(self, spec, spec_freqs, frame_rate):
        self.spec = spec
        self.spec_freqs = spec_freqs
        self.frame_rate = frame_rate
        self.frame_count = self.spec.shape[0]
        self.duration = self.frame_count / self.frame_rate

        img = util.lerp(self.spec, 0, 1, 0, 0xFF, clip=True).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 4, 2)
        img = img.transpose(1, 0, 2)
        self.spec_img = QtGui.QImage(img.tobytes(), img.shape[1], img.shape[0], QtGui.QImage.Format_RGB32)
        self.spec_img.save(os.path.join(os.path.dirname(__file__), 'spec.png'))
        self.spec_pm = QtGui.QPixmap.fromImage(self.spec_img)

        self.spec_pm_item = SpectrogramViewer.SpecGraphicsItem(self.spec_pm, self, self.scene)

        self.time_line = self.scene.addLine(0, 0, 0, self.spec_img.height(), pen=QtGui.QPen(QtGui.QColor.fromHsvF(0, 1, 1), 1.5))

        # self.value_label = QtGui.QGraphicsSimpleTextItem(self.scene)
        # self.scene.addItem(self.value_label)

        # self.translate(-self.spec.shape[0] / 2, 0)

    def update_time(self, time):
        self.time_line.setX(util.lerp(time, 0, self.duration, 0, self.spec_img.width()))
        self.view.centerOn(self.time_line)

    class SpecGraphicsItem(QtGui.QGraphicsPixmapItem):
        def __init__(self, pixmap, viewer, scene):
            super().__init__(pixmap, None, scene)

            self.viewer = viewer
            self.setAcceptHoverEvents(True)
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        def hoverMoveEvent(self, event):
            time_idx = int(event.pos().x())
            freq_idx = int(event.pos().y())
            value = self.viewer.spec[time_idx, freq_idx]
            time = util.lerp(time_idx, 0, self.viewer.frame_count, 0, self.viewer.duration)
            freq = self.viewer.spec_freqs[freq_idx]
            self.viewer.value_label.setText('{:.4g} Hz at {:.3f} secs: {:.3f}'.format(freq, time, value))

class PixelViewer(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        scene = QtGui.QGraphicsScene()
        scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

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
