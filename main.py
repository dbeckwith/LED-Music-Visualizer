import os
import time
import signal
import numpy as np

import config
import util
from visualizer import Visualizer
from audio import Audio
from gui import gui


def split_spec(spec, n):
    ranges = np.linspace(0, spec.shape[0], config.NUM_LED_CHANNELS + 1, dtype=np.int_)
    for i in range(config.NUM_LED_CHANNELS):
        spec_from = ranges[i]
        spec_to = ranges[i + 1]
        spec_range = spec_to - spec_from
        channel_vals = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, spec_to - spec_from), spec[spec_from:spec_to])
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
    parser = argparse.ArgumentParser(description='LED Music Visualizer by Daniel Beckwith')
    parser.add_argument('-t', '--test_mode', action='store_true', help='Test mode. Don\'t connect to external LED setup.')
    parser.add_argument('-b', '--brightness', type=float_normal, default=1.0, help='LED brightness factor from 0 to 1.')
    parser.add_argument('-a', '--audio_path', type=file, required=False, help='Path to MP3 file to use for visualization. If not given, a test signal is used.')
    parser.add_argument('-v', '--volume', type=float_normal, default=0.5, help='Music volume from 0 to 1.')
    parser.add_argument('-d', '--show_debug_window', action='store_true', help='Show the debug window.')
    args = parser.parse_args()

    gui.setup()

    with Visualizer(use_leds=not args.test_mode, brightness=args.brightness) as vis:
        with Audio(args.audio_path, audio_volume=args.volume, spectrogram_width=int(config.NUM_LEDS / 2 * config.NUM_LED_CHANNELS)) as audio:
            def sigint(signum, frame):
                vis.stop()
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGINT, sigint)
            gui.on_close = lambda: vis.stop()

            gui.start(args.show_debug_window)

            vis.start()
            audio.start()

            frames = 0

            util.timer('Running visualization')

            while vis.running:
                t = audio.elapsed_time

                pixels = np.zeros((config.NUM_LEDS, config.NUM_LED_CHANNELS), dtype=np.float64)
                spec = audio.spectrogram(t)

                low, mid, hi = tuple(split_spec(spec, config.NUM_LEDS // 2))
                # TODO: map to other hues besides RGB? (might need to be linearly independent)
                pixels[:, 0] = np.concatenate((low[::-1], low))
                pixels[:, 1] = np.concatenate((mid[::-1], mid))
                pixels[:, 2] = np.concatenate((hi[::-1], hi))
                # pixels[:, 0] = np.interp(np.linspace(0, 1, config.NUM_LEDS), np.linspace(0, 1, len(spec)), spec)

                # TODO: need to make less jumpy and increase contrast (as in make changes more pronounced)

                vis.send_pixels(pixels)
                frames += 1

                if t > 0: gui.update_fps(frames / t)
                gui.update_time(t)
                gui.update_leds(pixels)
                gui.update_spec(spec)

                gui.app.processEvents()

                # TODO: if fps too high, graph won't update
                if args.test_mode:
                    time.sleep(0.01)

                if not audio.running:
                    vis.stop()
                
    gui.stop()
    util.timer()

    import pyqtgraph
    pyqtgraph.exit()
