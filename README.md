# Ariana Grande Lyrics Generator
*Machine Learning Test*

Ariana Grande Lyrics Generator, or Ariana Generator for short, uses a Recurrent Neural Network to generate Ariana-Grande style song lyrics, based on all songs on the standard editions of her five albums: *thank u, next*, *Sweetener*, *Dangerous Woman*, *My Everything*, and *Yours Truely*. Originally, lyrics were restricted to her two most recent albums (*thank u, next* and *Sweetener*), because her style has evolved a lot over time, but more data was required to properly train the model.

Ariana Generator has 2 parts: the TensorFlow Keras RNN model builder, written in python, and the web interface, using TensorFlow.js.

## Python Model Builder
There are four files in the model builder:
```
ariana_model_builder.py
ariana_model_runner.py
ariana_model_saver.py
lyrics.txt
```
`lyrics.txt` contains the lyrics from all 5 of Ariana's albums.

`ariana_model_builder.py` loads the text from lyrics.txt and trains an RNN to produce similar lyrics.

`ariana_model_runner.py` loads the model weights and runs the model from the command line.

`ariana_model_saver.py` loads the model weights and the lyrics and saves the model so that it can be used for TensorFlow.js.

All of the python files are intended to be run from the command line, either by hand or from npm scripts in the web interface portion. They are not run from a server. The generation is done on-device using TensorFlow.js in the web interface.

## Web Interface
The Web Interface allows anyone without understanding of git, python, or tensorflow to use the AI. Simply visit the page, press the `Generate` button, and wait while it generates. Then it's listed right there.

