#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import signal
import traceback

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

    display = Display(brightness=args.brightness) if not args.test_mode else None
    audio = Audio(args.audio_path, audio_volume=args.volume)
    animation = Animation(audio.samples, audio.sample_rate)

    def sigint(signum, frame):
        audio.stop()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGINT, sigint)
    gui.on_close = lambda: audio.stop()

    gui.spec_viewer.set_spectrogram(animation.spec, animation.spec_freqs, animation.frame_rate)

    gui.start(args.show_debug_window)

    if display: display.start()
    audio.start()

    frame_times = []

    util.timer('Running visualization')

    try:
        while audio.running:
            t = audio.elapsed_time

            frame_timer = time.time()
            frame_times.append(frame_timer)
            while frame_timer - frame_times[0] >= 1.0:
                frame_times.pop(0)

            pixels = animation.get_frame(t)

            if display: display.send_pixels(pixels)

            gui.update_fps(len(frame_times))
            if not audio.is_paused():
                gui.update_time(t)
            gui.update_pixels(pixels)

            gui.app.processEvents()

            if gui.pause_requested:
                gui.pause_requested = False
                if not audio.is_paused():
                    audio.pause()
                else:
                    audio.unpause()

            if display and not display.running:
                break
    except:
        traceback.print_exc()
    finally:
        audio.stop()
        if display: display.stop()    
        gui.stop()

        util.timer()

        import pyqtgraph
        pyqtgraph.exit()
