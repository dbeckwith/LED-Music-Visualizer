import time
import signal
import numpy as np

import config
from visualizer import Visualizer


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
        start_time = time.time()
        last_fps_print_time = start_time
        start_frames = 0
        frames = 0
        offsets = np.array([0, -TAIL_LEN*1.5, -TAIL_LEN*3.0], dtype=np.float32)
        signal.signal(signal.SIGINT, lambda *args: vis.stop())
        while not vis.stopped:
            t = time.time()
            total_time = t - start_time

            pixels = np.zeros((config.NUM_LEDS, config.CHANNELS), dtype=np.float32)
            for c in range(config.CHANNELS):
                head = trail_pos(t + offsets[c])
                tail = trail_pos(t + offsets[c] - TAIL_LEN)
                if head <= tail:
                    min_x = head
                    max_x = tail
                    min_fade = 1.0
                    max_fade = 0.0
                else:
                    min_x = tail
                    max_x = head
                    min_fade = 0.0
                    max_fade = 1.0
                pixels[min_x:max_x+1, c] = np.linspace(min_fade, max_fade, num=max_x+1-min_x) ** 1
            vis.send_pixels(pixels * 1.0)

            frames += 1
            fps_print_time = t - last_fps_print_time
            if fps_print_time >= FPS_PRINT_INTERVAL:
                total_frames = frames - start_frames
                print('{:d} frames in {:.2f} seconds ({:.2f} fps)'.format(total_frames, fps_print_time, total_frames / fps_print_time))
                last_fps_print_time = t
                start_frames = frames
        
        signal.signal(signal.SIGINT, signal.SIG_DFL)
