# Machine Learning Engineer Nanodegree
## Capstone Project - Hand Digit Recognizer via Neural Network
### Install
In order to run the juypter notebook comitted in this repository, Python 3.6 version must be installed. The following libraries need to be installed as well:
* [sklearn](https://scikit-learn.org/stable/)
* [scikit-image](https://scikit-image.org/)

To run the notebooks, you need to have [Jupyter](https://jupyter.org/) installed. If you are using Windows, it is recommended that you install [Anaconda](https://www.anaconda.com/) to run juypter. If you are using Mac or Ubuntu, [follow this instruction](https://jupyter.org/install) to install Jupyter. 

### Code
All the necesary code is provided in `handwritten-digit-recognition.ipynb` notebook file. The handwritten digit images are already committed to this; however, you maybe able to download the MNIST dataset manually through this [link](http://yann.lecun.com/exdb/mnist/). 

### Run
In order to run the jupyter notebook, open up a terminal, navigate into the top-level directory of this repository and type the following commands:
```
ipython notebook handwritten-digit-recognition.ipynb
```


### Data
There are two different sources to download the data:
* [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
* [Kaggle dataset](https://www.kaggle.com/c/digit-recognizer/data)

Download the dataset and uncompress it into the `data` directory. The name of the csv file should be `data/test.csv`.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
Visually, if we omit the "pixel" prefix, the pixels make up the image like this:
```
000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 
```