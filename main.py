#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import signal

import config
import util
from display import Display
from audio import Audio
from animation import Animation
from gui import gui


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

    display = Display(use_leds=not args.test_mode, brightness=args.brightness)
    audio = Audio(args.audio_path, audio_volume=args.volume)
    animation = Animation(audio.samples, audio.sample_rate)

    def sigint(signum, frame):
        display.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGINT, sigint)
    gui.on_close = lambda: display.stop()

    gui.start(args.show_debug_window)

    display.start()
    audio.start()

    frames = 0

    util.timer('Running visualization')

    while display.running:
        t = audio.elapsed_time

        pixels = animation.get_frame(t)

        display.send_pixels(pixels)
        frames += 1

        if t > 0: gui.update_fps(frames / t)
        gui.update_time(t)
        gui.update_pixels(pixels)
        gui.update_spec(animation.get_spec_frame(t))

        gui.app.processEvents()

        # TODO: if fps too high, graph won't update
        if args.test_mode:
            time.sleep(0.01)

        if not audio.running:
            display.stop()

    audio.stop()
                
    gui.stop()
    util.timer()

    import pyqtgraph
    pyqtgraph.exit()
