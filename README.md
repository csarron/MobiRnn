
# MobiRNN for EMDL17

`Data` folder has a zip file which needs to be unzipped and placed on the sdcard of Android smartphone.

`Model` folder contains Python code for LSTM model on smartphone sensor data using default configuration (2 layer, 32 hidden units). Other layer and hidden units configurations can be found at different tags [in this repo](https://github.com/csarron/lstm_har). For example, 3 layer 128 units, see tag `v3.128`

Android implementation of MobiRNN is in the `Code` folder. Cuda-based version can be found at the `cuda` branch of [this repo](https://github.com/csarron/MobiRnn), where `util` branch takes GPU utilization into account.


## Acknowledgement
The original model source code is modified from
[guillaume-chevalier](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)
