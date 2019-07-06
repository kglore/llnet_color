# LLNet: Low-light Image Enhancement with Deep Learning (Color)
_By: Kin Gwn Lore, Adedotun Akintayo, Soumik Sarkar_

This is the **official** repository which includes the codes and modules used for running color-LLNet via a Graphical User Interface. Users can choose to train the network from scratch, or to enhance multiple images using a specific trained model. Note that we had hosted on BitBucket before, but those codebase is now severely outdated. Please use GitHub (this repository) instead.

The project code has been updated in 2016 since the publication of ["LLNet: A deep autoencoder approach to natural low-light image enhancement"](https://www.sciencedirect.com/science/article/abs/pii/S003132031630125X) to support color version. The code is also used to generate results included in my [thesis](https://lib.dr.iastate.edu/etd/15965/). This code is no longer supported as Theano is also no longer supported. However, please feel free to try to run the code.

**NOTE: The model is created in the past but it was using packages which are now many, many versions behind. The current commit (as of June 2019) converts the model object into weights and biases only. The performance of the model may be impacted as a result of numeric/version incompatibilities.**

No license has been created for the code. Please contact the authors if you wish to use the code for other than academic purposes. Thanks!

![alt text](https://github.com/kglore/llnet_color/blob/master/readme/structure.png)

## Requirements

### Software requirements
- Anaconda with Python 2.7
- Theano 0.8
- CUDA-enabled device to unpickle trained model
- Easygui: Simple GUI for training/inference

Example installation commands. Please look at req.txt for full package list of the environment configs used to run the code:
```
conda create -n llnet --file req.txt
source activate llnet
pip install easygui
```

You may need to configure your .theanorc file in the home directory (Theano configurations):
```
[global]
device=cuda
#device=gpu
#device=cpu
floatX=float32
exception_verbosity=high
#lib.cnmem=0.85
gpuarray.preallocate=0.85

[nvcc]
fastmath = True
```

### Download the model
Please download the following object and place it in the 'model/' folder. Download model from:
- [Bitbucket](https://bitbucket.org/kglore/llnet-color/src/master/model/003_model.obj) (Obsolete! Will NOT work)
- [Dropbox](https://www.dropbox.com/s/hxwvxvngqs0j1xj/003_model.wgt?dl=0)

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
