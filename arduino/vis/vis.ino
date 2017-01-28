#include <Adafruit_NeoPixel.h>

#define STRIP_PIN 4
#define LED_COUNT 60
//#define BUF_LEN LED_COUNT*4

Adafruit_NeoPixel strip = Adafruit_NeoPixel(LED_COUNT, STRIP_PIN, NEO_GRB | NEO_KHZ800);

//byte buf[BUF_LEN];
//byte *buf_pos;
//int bytes_read;
//int extra;
//int i;
byte index, red, green, blue;

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.show();
//  buf_pos = buf;
//  extra = 0;
}

void loop() {
//  bytes_read = Serial.readBytes(buf_pos, BUF_LEN - extra);
//  for (i = 0; i < bytes_read; i += 4) {
//    strip.setPixelColor(buf[i], buf[i+1], buf[i+2], buf[i+3]);
//  }
//  extra = bytes_read - (i - 4);
//  for (i = 0; i < extra; i++) {
//    buf[i] = buf[bytes_read - extra + i];
//  }
//  buf_pos = buf + extra;
//  strip.show();
  while (true) {
    while (!Serial.available());
    index = Serial.read();
    while (!Serial.available());
    red = Serial.read();
    while (!Serial.available());
    blue = Serial.read();
    while (!Serial.available());
    green = Serial.read();
    strip.setPixelColor(index, red, green, blue);
    if (index == LED_COUNT - 1) {
      strip.show();
  //    delay(50);
      Serial.write(0x00);
    }
  }
}

