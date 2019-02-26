# LLNet: Low-light Image Enhancement with Deep Learning (Color)#

This repository includes the codes and modules used for running color-LLNet via a Graphical User Interface. Users can choose to train the network from scratch, or to enhance multiple images using a specific trained model.

The project code has been updated since the publication of "LLNet: A deep autoencoder approach to natural low-light image enhancement" to support color version.

This code is no longer supported as Theano is also no longer supported. However, please feel free to try to run the code.

No license has been created for the code. Please contact the authors if you wish to use the code for other than academic purposes. Thanks!

## How do I run the program? ##

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

Datasets need to be saved as .MAT file with the '-v7.3' tag in MATLAB. The saved variables are:

```
train_set_x     (N x whc)   Noisy, darkened training data
train_set_y     (N x whc)   Clean, bright training data
valid_set_x     (N x whc)   Noisy, darkened validation data
valid_set_y     (N x whc)   Clean, bright validation data
test_set_x      (N x whc)   Noisy, darkened test data
test_set_y      (N x whc)   Clean, bright test data
```

Where N is the number of examples and w, h are the width and height of the patches, respectively. C is the number of channels (i.e. 3 for RGB images). R vectors, G vectors, and B vectors are horizontally concatenated. Test data are mostly used to plot the test patches; in actual applications we are interested to enhance a single image. Use the test command instead.

## Citation ##

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