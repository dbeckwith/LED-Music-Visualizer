# LED Music Visualizer :bulb::musical_note::bulb:

I created this project as part of my Humanities & Arts Practicum at Worcester Polytechnic Institute. The practicum was called "[Light Art](http://www.joshuarosenstock.com/teaching/lightart-c17/)" and was directed by [Prof. Joshua Rosenstock](http://www.joshuarosenstock.com/).

The goal behind the project was to try to find a way to map the sounds we hear and the feelings evoked in us by music to light. I wanted to try to use light to convey the same emotions that I was feeling from listening to a song.

## Installation

### Hardware

You'll need to aquire the following hardware in order to build the visualizer, all available from [Adafruit](https://www.adafruit.com/):

* [FadeCandy](https://www.adafruit.com/product/1689)
* [Mini USB cable](https://www.adafruit.com/products/260)
* [5V 10A power supply](https://www.adafruit.com/product/658)
* [DC barrel jack](https://www.adafruit.com/product/373)
* [3-pin JST cables](https://www.adafruit.com/products/1663)
* [NeoPixel RGB LED strip](https://www.adafruit.com/products/1138)

Follow Adafruit's excellent [tutorial](https://learn.adafruit.com/led-art-with-fadecandy?view=all#wiring-your-leds) on how to wire the FadeCandy. The only difference is that you will be wiring an LED strip, not an LED matrix.

### Software

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

Next, open [`config.py`](config.py) and make sure that the values there match your setup. `PIXEL_COUNT` should be the number of NeoPixels on your LED strip, `CHANNELS_PER_PIXEL` should be `3` for the RGB pixels, and `FADECANDY_HOST` and `FADECANDY_PORT` should match the values shown when the FadeCandy server was launched.

Next, run the main script from the repo's top level directory to get help on its usage:

```bash
python main.py -h
```
