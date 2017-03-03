# LED Music Visualizer :bulb::musical_note::bulb:

I created this project as part of my Humanities & Arts Practicum at Worcester Polytechnic Institute. The practicum was called "[Light Art](http://www.joshuarosenstock.com/teaching/lightart-c17/)" and was directed by [Prof. Joshua Rosenstock](http://www.joshuarosenstock.com/).

The goal behind the project was to try to find a way to map the sounds we hear and the feelings evoked in us by music to light. I wanted to try to use light to convey the same emotions that I was feeling from listening to a song.

## How It Works

First, the program loads the given audio file and does some pre-processing on it. The pre-processing involves creating a [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) of the audio using the [Discrete Fourier Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) and applying a [Mel Filter](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/). Using this high-level information about the component frequencies of the audio, the program generates an animation. The animation is drawn for each frame on a canvas in memory, mapping different frequencies to different properties of the "blips" that are drawn. Then each frame is converted into RGB pixel data that is sent to a [FadeCandy LED controller](https://www.adafruit.com/product/1689), which controls an LED matrix that displays the animation. The animation is partly reactive to the spectrogram, and partly hand-choreographed. The audio file is also played along with the animation to make sure that the two are perfectly synchronized.

## Demo

[Demo video](https://vimeo.com/206593451)

[Blog post](http://www.joshuarosenstock.com/teaching/lightart-c17/led-music-visualizer-final/)

## Try It Out

### Hardware

I used all the same materials from Adafruit's excellent [tutorial](https://learn.adafruit.com/led-art-with-fadecandy?view=all) on how to use their FadeCandy LED controller board. Simply follow this tutorial and you'll have the same electronics set-up I used. The only other material I used was the translucent white acrylic from [Delvie's Plastics](http://www.delviesplastics.com/p/Translucent_Cast_Acrylic_Sheet.html), which I cut to make an enclosure around the LED matrix. This allows the piercing, direct light of the LEDs to be diffused and softend a bit, which creates a more pleasing display.

### Software

At the moment, the visualization is specifically coreographed to the song ["Such Great Heights" by The Postal Service](https://www.youtube.com/watch?v=FatiC6NxCcE), so you'll need an MP3 file of that song.

You'll need to install the following software in order to run the code:

* [Python 3.5](https://www.python.org/downloads/)
* [ffmpeg 3.2.2](https://ffmpeg.org/download.html) (only a "static" build with the executables is required)
* [FadeCandy Server 02](https://github.com/scanlime/fadecandy/releases/tag/package-02)

Make sure Python and ffmpeg are in your system path. Then, clone this repo or download and unzip the code. In the top directory, use the following command to install the required Python libraries with pip:

```bash
python -m pip install -r requirements.txt
```

You may want to do this in a [Virtual Environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to avoid library version conflicts.

## Usage

First, launch the FadeCandy server by running the appriopriate `fcserver` executable for your platform in the `bin` folder of the FadeCandy download. Once the server is running, connect the FadeCandy board by USB to your computer. You should see a message on the server saying it has connected to a board. If it's your first time connecting a FadeCandy board, you may have wait a minute for your OS to install the drivers.

Next, open [`config.py`](config.py) and make sure that the values there match your setup. `DISPLAY_SHAPE` should be a tuple of the number of columns and rows in your LED matrix (probably just 8 by 8), `CHANNELS_PER_PIXEL` should be `3` for the RGB pixels, and `FADECANDY_HOST` and `FADECANDY_PORT` should match the values shown when the FadeCandy server was launched.

Next, run the main script from the repo's top level directory to get help on its usage:

```bash
python main.py -h
```

At the moment, the visualization is specifically coreographed to the song ["Such Great Heights" by The Postal Service](https://www.youtube.com/watch?v=FatiC6NxCcE), so if you own an MP3 of that song, supply a path to it for the `--audio-path` argument.
