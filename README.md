# RBM_DBN
Experimenting with RBMs using scikit-learn on MNISTand simulating a DBN using Keras.

MNIST data has been used for these experiments. The compressed files can be downloaded [here](http://yann.lecun.com/exdb/mnist/). 

You can use [python-mnist](https://pypi.python.org/pypi/python-mnist/) if you're finding it hard to parse and process the data. 

## Dependencies

- numpy==1.12.0
- keras==2.0.6
- scikit_learn==0.19b2


## Files:

1. RBM.ipynb - Jupyter Notebook containing experiments and results done on the mnist data, I have included comments wherever necessary
2. RBM.html - The above file in HTML format for quick viewing
3. DBN.py - A DBN wrapper simulated using RBMs in scikit-learn followed by Keras. Weights learnt during RBM training are then initialised in a MLP built using Keras Sequential Model. Inline documentation and example code are included.
