import os
import time
import signal
import numpy as np

import config
import util
from visualizer import Visualizer
from audio import Audio


def trail_pos(t):
    return int(np.clip((np.sin(t * 2 * np.pi / 4) + 1) / 2 * config.NUM_LEDS, 0, config.NUM_LEDS-1))
    # return int(np.clip(2 * abs((t % 4) / 4 * config.NUM_LEDS - config.NUM_LEDS / 2), 0, config.NUM_LEDS-1))

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
    args = parser.parse_args()

    if not args.music_path:
        args.music_path = ':debug:'

    with Visualizer(use_arduino=args.use_arduino, brightness=args.brightness) as vis:
        with Audio(args.music_path, audio_volume=args.volume) as audio:
            offsets = np.array([0, -TAIL_LEN*1.5, -TAIL_LEN*3.0], dtype=np.float32)

            def sigint(signum, frame):
                vis.stop()
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGINT, sigint)

            vis.start()
            audio.start()

            util.timer('Running visualiztion')

            while vis.running:
                pixels = np.zeros((config.NUM_LEDS, config.NUM_LED_CHANNELS), dtype=np.float64)
                spec = audio.spectrogram(audio.elapsed_time)
                ranges = np.linspace(0, spec.shape[0], config.NUM_LED_CHANNELS + 1, dtype=np.int_)
                for i in range(config.NUM_LED_CHANNELS):
                    spec_from = ranges[i]
                    spec_to = ranges[i + 1]
                    spec_range = spec_to - spec_from
                    # TODO: since less channel vals then in spec, need to mean over all the spec vals each channel val covers, not interp
                    channel_vals = np.interp(np.linspace(0, spec_range, config.NUM_LEDS // 2, endpoint=False), np.arange(spec_range), spec[spec_from:spec_to])
                    pixels[:, i] = np.concatenate((channel_vals[::-1], channel_vals))

                # pixels = np.zeros((config.NUM_LEDS, config.NUM_LED_CHANNELS), dtype=np.float32)
                # for c in range(config.NUM_LED_CHANNELS):
                #     head = trail_pos(audio.elapsed_time + offsets[c])
                #     tail = trail_pos(audio.elapsed_time + offsets[c] - TAIL_LEN)
                #     if head <= tail:
                #         min_x = head
                #         max_x = tail
                #         min_fade = 1.0
                #         max_fade = 0.0
                #     else:
                #         min_x = tail
                #         max_x = head
                #         min_fade = 0.0
                #         max_fade = 1.0
                #     pixels[min_x:max_x+1, c] = np.linspace(min_fade, max_fade, num=max_x+1-min_x) ** 1

                vis.send_pixels(pixels)

                if not audio.running:
                    vis.stop()
                    
    util.timer()
