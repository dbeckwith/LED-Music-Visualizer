import time
import numpy as np

from com import Com


if __name__ == '__main__':
    with Com() as com:
        max_time = 5
        print('Timing for {:.2f} seconds...'.format(max_time))
        start_time = time.time()
        frames = 0
        xs = np.zeros((10,), dtype=np.int32)
        xs[:] = 30
        while True:
            # pixels[:, 0] = np.roll(np.linspace(0.0, 1.0, num=60), frames)
            xs = np.roll(xs, 1)
            xs[0] = (np.sin(frames / 20) + 1) / 2 * 60
            if xs[0] <= xs[-1]:
                min_x = xs[0]
                max_x = xs[-1]
                min_fade = 1.0
                max_fade = 0.0
            else:
                min_x = xs[-1]
                max_x = xs[0]
                min_fade = 0.0
                max_fade = 1.0
            pixels = np.zeros((60, 3), dtype=np.float32)
            pixels[min_x:max_x+1, 0] = np.linspace(min_fade, max_fade, num=max_x+1-min_x)
            com.send_pixels(pixels * 0.1)
            frames += 1
            total_time = time.time() - start_time
            if total_time >= max_time: break
        print('{:d} frames in {:.2f} seconds ({:.2f} fps)'.format(frames, total_time, frames / total_time))
