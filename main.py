import time
import signal
import numpy as np

import config
from visualizer import Visualizer
from audio import Audio


def trail_pos(t):
    return int(np.clip((np.sin(t * 2 * np.pi / 4) + 1) / 2 * config.NUM_LEDS, 0, config.NUM_LEDS-1))
    # return int(np.clip(2 * abs((t % 4) / 4 * config.NUM_LEDS - config.NUM_LEDS / 2), 0, config.NUM_LEDS-1))

TAIL_LEN = 0.5
FPS_PRINT_INTERVAL = 5.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--use_arduino', action='store_true')
    args = parser.parse_args()

    with Visualizer(use_arduino=args.use_arduino, brightness=0.2) as vis:
        # TODO: try a simpler sound, like a sine wave with frequency going in a predictable pattern
        with Audio(r"C:\Users\Daniel\Dropbox\Music\music\Deadmau5\Random Album Title\1-11 Arguru.mp3", frame_size=0.1, audio_volume=0.5) as audio:
            offsets = np.array([0, -TAIL_LEN*1.5, -TAIL_LEN*3.0], dtype=np.float32)

            def sigint(signum, frame):
                vis.stop()
                signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGINT, sigint)

            start_time = time.time()
            vis.start()
            audio.start()

            while vis.running:
                t = time.time()
                total_time = t - start_time

                pixels = np.zeros((config.NUM_LEDS, config.CHANNELS), dtype=np.float32)

                pixels[:, 0] = audio.spectrogram(total_time)

                # for c in range(config.CHANNELS):
                #     head = trail_pos(t + offsets[c])
                #     tail = trail_pos(t + offsets[c] - TAIL_LEN)
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
