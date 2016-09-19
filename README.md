# python-image-object-detection
Package that wraps third party image object libraries - research for 
Newslinn http://www.newslinn.com

# PURPOSE & GOALS

## Purpose
To be a package within a bigger system. This is designed around extracting as much
object data and image data from a photograph. It's not concerned with detection
or recognition. As it's purpose is to aid in investigating corrolations with object data as
part of image classification.

eg. if an image has high amounts of objects vs. one with low amounts, which image
is typically more classified as valid?

The end result from this package is to include the abstract data as factors within
a classifier. 

## Goal
To determine if the abstract data in an image aids in classification.

## Ideas
- parse an image and extract as much object data as possible
- ie. instead of calling out to a API?
- how many edges? (high edges = high detail, city or crowd scene?)
- taken during day or night
- how much pixels in foreground vs background
- color mean

# Newslinn
Newslinn www.newslinn.com is a citizen-to-journalist communication 
network. At the core of the network is verification technology that uses
artificial intelligence and machine learning to classify if 
communication is fraud or spam.

eg. if a citizen protester sends a photo to a journalist. Was that photo
edited in photoshop? or is that photo from 6 years ago and taken in 
another location etc. etc.

# REFERENCES


### Shi-Tomasi Corner Detector & Good Features to Track
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html#shi-tomasi


### Changing Color-space to HSV 
http://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html



### SURF in Python
https://gist.github.com/moshekaplan/5106221
















## OTHER USEFUL ITEMS

### Unofficial Windows Binaries for Python Extension Packages
http://www.lfd.uci.edu/~gohlke/pythonlibs/

# you will need to install opencv seperately
# in particular, "python open cv"
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
# and typically the 32bit version of it
# also make sure to download the right version
# via the "cp34" "cp35" "cp27" cp = cpython version

### Cookiecutter for initial project files and configuration
https://github.com/audreyr/cookiecutter-pypackage


