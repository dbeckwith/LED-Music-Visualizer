import os
import time
import signal
import numpy as np

import config
import util
from visualizer import Visualizer
from audio import Audio
from gui import gui


def resample(x, n):
    if n == x.shape[0]:
        return x.copy()
    if n > x.shape[0]:
        return np.interp(np.linspace(0, x.shape[0], n, endpoint=False), np.arange(x.shape[0]), x)
    raise NotImplementedError
    # TODO: upsampling
    coords = np.linspace(0, x.shape[0], n + 1, endpoint=True)
    for i in range(n):
        coord_from = coords[i]
        coord_to = coords[i + 1]

def split_spec(spec, n):
    ranges = np.linspace(0, spec.shape[0], config.NUM_LED_CHANNELS + 1, dtype=np.int_)
    for i in range(config.NUM_LED_CHANNELS):
        spec_from = ranges[i]
        spec_to = ranges[i + 1]
        spec_range = spec_to - spec_from
        channel_vals = resample(spec[spec_from:spec_to], n)
        yield channel_vals


TAIL_LEN = 0.5
FPS_PRINT_INTERVAL = 5.0

if __name__ == '__main__':
    def file(path):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise ValueError('{} is not a valid file'.format(path))
        return path

    def float_normal(x):
        x = float(x)
        if not (0 <= x <= 1):
            raise ValueError('must be between 0 and 1')
        return x

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--use_arduino', action='store_true')
    parser.add_argument('-m', '--music_path', type=file, required=False)
    parser.add_argument('-b', '--brightness', type=float_normal, default=1.0)
    parser.add_argument('-v', '--volume', type=float_normal, default=0.5)
    parser.add_argument('-d', '--show_debug_window', action='store_true')
    args = parser.parse_args()

    with Visualizer(use_arduino=args.use_arduino, brightness=args.brightness) as vis:
        with Audio(args.music_path, audio_volume=args.volume) as audio:
            def sigint(signum, frame):
                vis.stop()
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGINT, sigint)
            gui.view.closeEvent = lambda *args: vis.stop()

            gui.start(args.show_debug_window)

            vis.start()
            audio.start()

            frames = 0

            util.timer('Running visualization')

            while vis.running:
                t = audio.elapsed_time

                pixels = np.zeros((config.NUM_LEDS, config.NUM_LED_CHANNELS), dtype=np.float64)
                spec = audio.spectrogram(t)
                # TODO: might be wrong about if low is lower part or higher part?? need to try with test signal
                low, med, hi = tuple(split_spec(spec, config.NUM_LEDS // 2))
                # TODO: map to other hues besides RGB? (might need to be linearly independent)
                pixels[:, 0] = np.concatenate((low[::-1], low))
                pixels[:, 1] = np.concatenate((hi[::-1], hi))
                pixels[:, 2] = np.concatenate((med, med[::-1]))

                vis.send_pixels(pixels)
                frames += 1

                gui.update_fps(frames / t)
                gui.update_time(t)
                gui.update_leds(pixels)
                gui.update_spec(spec)

                gui.app.processEvents()

                # TODO: if fps too high, graph won't update
                if not args.use_arduino:
                    time.sleep(0.01)

                if not audio.running:
                    vis.stop()
                
    gui.stop()
    util.timer()

    import pyqtgraph
    pyqtgraph.exit()
