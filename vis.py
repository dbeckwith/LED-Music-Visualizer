import time
import numpy as np

import config
from com import Com


def pos(f):
    return np.clip(int((np.sin(f * 2 * np.pi / config.APPROX_FPS / 4) + 1) / 2 * config.NUM_LEDS), 0, config.NUM_LEDS-1)


TAIL_LEN = 10
FPS_PRINT_INTERVAL = 5.0

if __name__ == '__main__':
    with Com(debug_mode=True) as com:
        start_time = time.time()
        start_frames = 0
        frames = 0
        offsets = np.array([0, -config.APPROX_FPS/2, -config.APPROX_FPS], dtype=np.int32)
        xs = np.zeros((TAIL_LEN, config.CHANNELS), dtype=np.int32)
        for c in range(config.CHANNELS):
            xs[:, c] = pos(offsets[c])
        while True:
            pixels = np.zeros((config.NUM_LEDS, config.CHANNELS), dtype=np.float32)
            xs = np.roll(xs, 1, axis=0)
            for c in range(config.CHANNELS):
                xs[0, c] = pos(frames + offsets[c])
                if xs[0, c] <= xs[-1, c]:
                    min_x = xs[0, c]
                    max_x = xs[-1, c]
                    min_fade = 1.0
                    max_fade = 0.0
                else:
                    min_x = xs[-1, c]
                    max_x = xs[0, c]
                    min_fade = 0.0
                    max_fade = 1.0
                pixels[min_x:max_x+1, c] = np.square(np.linspace(min_fade, max_fade, num=max_x+1-min_x))
            com.send_pixels(pixels * 1.0)
            frames += 1
            t = time.time()
            total_time = t - start_time
            if total_time >= FPS_PRINT_INTERVAL:
                total_frames = frames - start_frames
                print('{:d} frames in {:.2f} seconds ({:.2f} fps)'.format(total_frames, total_time, total_frames / total_time))
                start_time = t
                start_frames = frames
