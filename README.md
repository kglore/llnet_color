# LLNet: Low-light Image Enhancement with Deep Learning (Color)

This repository includes the codes and modules used for running color-LLNet via a Graphical User Interface. Users can choose to train the network from scratch, or to enhance multiple images using a specific trained model.

The project code has been updated since the publication of "LLNet: A deep autoencoder approach to natural low-light image enhancement" to support color version.

This code is no longer supported as Theano is also no longer supported. However, please feel free to try to run the code.

No license has been created for the code. Please contact the authors if you wish to use the code for other than academic purposes. Thanks!

## Requirements

### Software requirements
Anaconda 2.7
Theano 0.9
CUDA-enabled device to unpickled trained model

### Download the model
The model is hosted over at BitBucket (our older repository). Please download the following object and place it in the 'model/' folder:
https://bitbucket.org/kglore/llnet-color/src/master/model/003_model.obj

## How do I run the program?

Open the terminal and navigate to this directory. Type:
```
#!bash
python llnet_color.py
```
to launch the program with GUI. For command-line only interface, you type the following command in the terminal.

To train a new model, enter:
```
#!bash
python llnet_color.py train [TRAINING_DATA]
```

To enhance an image, enter:
```
#!bash
python llnet_color.py test [IMAGE_FILENAME] [MODEL_FILENAME]
```

For example, you may type:
```
#!bash
python llnet_color.py train datafolder/yourdataset.mat
python llnet_color.py test somefolder/darkpicture.png models/save_model.obj
```
where file names do not need to be in quotes.

Datasets need to be saved as .MAT file with the '-v7.3' tag in MATLAB. However, it is actually a HDF5 file and can also be created with Python's H5PY module. When writing into HDF5, we expect matrix of shape (dimensions, samples) as opposed to the regular (samples, dimension). However, when data goes into the neural network it should have the following shape:
```
train_set_x     (N x whc)   Noisy, darkened training data
train_set_y     (N x whc)   Clean, bright training data
valid_set_x     (N x whc)   Noisy, darkened validation data
valid_set_y     (N x whc)   Clean, bright validation data
test_set_x      (N x whc)   Noisy, darkened test data
test_set_y      (N x whc)   Clean, bright test data
```
Where N is the number of examples and w, h are the width and height of the patches, respectively. C is the number of channels (i.e. 3 for RGB images). R vectors, G vectors, and B vectors are horizontally concatenated. Test data are mostly used to plot the test patches; in actual applications we are interested to enhance a single image. Use the test command instead.

You can run model training over a dummy dataset of patch size 17x17 pixels, over 3 channels. The dummy data is just a random uniform noise.

## Results Sample
![alt text](https://github.com/kglore/llnet_color/blob/master/readme/samples.png)

## Citation

If you wish to cite the work, please use the following citation:
```
@article{lore2017llnet,
  title={LLNet: A deep autoencoder approach to natural low-light image enhancement},
  author={Lore, Kin Gwn and Akintayo, Adedotun and Sarkar, Soumik},
  journal={Pattern Recognition},
  volume={61},
  pages={650--662},
  year={2017},
  publisher={Elsevier}
}
```
