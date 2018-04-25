# Traffic Light Detection with YOLOv2

## YOLOv2 (You only look once)

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

Dataset: [LISA Traffic Light Dataset](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset)

YAD2K: [Folder used for retraining the model](https://github.com/allanzelener/YAD2K)

![Retrained YOLO model on the Traffic Light Dataset](out_day/dayClip6--00108.jpg)

--------------------------------------------------------------------------------

## Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/) (For Keras model serialization.)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)
- [pydot-ng](https://github.com/pydot/pydot-ng) (Optional for plotting model.)

--------------------------------------------------------------------------------

## More Details

`YAD2K` can be downloaded from the website mentioned above and was used to retrain
the YOLOv2 model.

`package_dataset.py` and `shuffle_data.py` can be used to package the data and prepare
it to be the input for the `retrain_yolo.py` from `YAD2K`.

`tl_detection.ipynb` is the notebook with the final implementation and evaluation code.

`yolo_utils` provides some useful functions that are used in the notebook.



--------------------------------------------------------------------------------
