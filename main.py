import time
import signal
import numpy as np

import config
from visualizer import Visualizer


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
        xs = ([], [], [])
        signal.signal(signal.SIGINT, lambda *args: vis.stop())
        while not vis.stopped:
            t = time.time()
            total_time = t - start_time

            pixels = np.zeros((config.NUM_LEDS, config.CHANNELS), dtype=np.float32)
            for c in range(config.CHANNELS):
                pos = int(np.clip(int((np.sin((t + offsets[c]) * 2 * np.pi / 4) + 1) / 2 * config.NUM_LEDS), 0, config.NUM_LEDS-1))
                xs[c].insert(0, (pos, t))
                while t - xs[c][-1][1] >= TAIL_LEN:
                    xs[c].pop()
                if xs[c][0][0] <= xs[c][-1][0]:
                    min_x = xs[c][0][0]
                    max_x = xs[c][-1][0]
                    min_fade = 1.0
                    max_fade = 0.0
                else:
                    min_x = xs[c][-1][0]
                    max_x = xs[c][0][0]
                    min_fade = 0.0
                    max_fade = 1.0
                pixels[min_x:max_x+1, c] = np.linspace(min_fade, max_fade, num=max_x+1-min_x) ** 5
            vis.send_pixels(pixels * 1.0)

            frames += 1
            fps_print_time = t - last_fps_print_time
            if fps_print_time >= FPS_PRINT_INTERVAL:
                total_frames = frames - start_frames
                print('{:d} frames in {:.2f} seconds ({:.2f} fps)'.format(total_frames, fps_print_time, total_frames / fps_print_time))
                last_fps_print_time = t
                start_frames = frames
        
        signal.signal(signal.SIGINT, signal.SIG_DFL)
