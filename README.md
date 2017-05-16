# MobiRnn on Android

##Intro
This repo is for running [LSTM model](https://github.com/csarron/lstm_har) on mobile devices. 
Currently we support the following mode:
- Plain CPU (Java)
- Native CPU (C)
- GPU (RenderScript)
- Eigen CPU (C++)
- TensorFlow CPU (Java)

## Usage
Just run `./gradlew iR` to install MobiRNN on your connected phone. 

You can run `./gradlew pu` to generate apk file or simply download the `blob/mobile-release.apk` in the repo.


