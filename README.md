# TensorFlow Quickstart

## Overview

Get started with TensorFlow quickly by following these steps.

## Installation

To use TensorFlow on Amazon AWS EC2 using a pre-configured AMI follow
[these steps](amazon-setup.md).

To install TensorFlow on your computer follow [these
steps](local-setup.md).

## Connect to EC2

If you are using an AWS instance, then connect to it using your PEM
file, like this.

    ssh -i PEM_FILE.pem ubuntu@EC2-INSTANCE.amazonaws.com

Replace `EC2-INSTANCE.amazonaws.com` with the actual instance name or
IP address.

## IPython

Once you are connected, start `ipython` as follows.

    CUDA_VISIBLE_DEVICES=1 ipython

The `CUDA_VISIBLE_DEVICES=1` is important as without it TensorFlow
will segfault on some EC2 instances.

This will be your IPython terminal.

## TensorFlow

Test that TensorFlow is installed. Run the following on the IPython
terminal.

```python
import tensorflow as tf

print tf.__version__

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
```

If TensorFlow does not work you can make Keras work with Theano
instead. To do this edit the `~/.keras/keras.json` file, and replace
`"tensorflow"` with `"theano"`.

## Keras

Test that Keras is installed.

```python
import keras.backend
print keras.backend._BACKEND
```

# Demo: Build AND/OR Network

## NOT

On paper build a 1-neuron network that implements the NOT function.
What should the incoming weight be?

## AND

On paper build a 2-layer 3-neuron network that implements the AND
function. What should the weights be?

## OR

On paper build a 2-layer 3-neuron network that implements the OR
function. What should the weights be?

# Demo: XOR

## XOR Using Keras

Save this code to a file.

Use `%load` to load this into `IPython`.

```python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

data = np.array([
  [0,0,0],
  [0,1,1],
  [1,0,1],
  [1,1,0]])
data.shape
X = data[:,0:2]
y = data[:,2]

model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X, y, batch_size=4, nb_epoch=2000, show_accuracy=True)
loss, accuracy = model.evaluate(X, y, show_accuracy=True, verbose=1)
print("Test fraction correct (Accuracy) = {:.2f}".format(accuracy))
print model.predict(X)
```

## Changing Neuron Type

In this section we will change hyperparameters and see if we can
optimize our network.

Change the neuron types to `tanh` and `relu`. 

Does the network converge faster?

Change batch sizes.

Does the network converge faster?

# Demo: Model Visualization

## Converting to PNG

Plot model as graph and save to file.

```python
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
```

`plot` optional arguments:

Argument       |Default  |Meaning
--------       |-------  |-------
`recursive`    |`True`   |Whether to recursively explore container layers
`show_shape`   |`False`  |Whether output shapes are shown in the graph

# Demo: Stocks

## Download Data

```bash
!curl 'http://real-chart.finance.yahoo.com/table.csv?s=AAPL&g=d&ignore=.csv' \
  > AAPL.csv
```

## Input Data

```python
import numpy as np
import theano

# Load prices
prices = np.loadtxt("AAPL.csv",
  dtype=theano.config.floatX,
  usecols=[1,2,3,4,5,6],
  delimiter=',', 
  skiprows=1)
prices.shape

# Reverse
prices = prices[::-1]

# Scale
prices = prices / prices.max(axis=0)

# Log
prices = np.log(prices + 1)

# Today's prices
prices_drop_last = prices[:-1]
prices_today = prices_drop_last

# Tomorrow's prices
prices_drop_first = prices[1:]
prices_tomorrow = prices_drop_first

# Check shapes
prices.shape
prices_today.shape
prices_tomorrow.shape

# Check prices
prices[:,-1]
prices_today[:,-1]
prices_tomorrow[:,-1]

# Does price tomorrow go up or down?
change = (prices_tomorrow[:,-1] - prices_today[:,-1]) > 0

# Converts np arrays to categorical
def np_to_categorical(arr):
    from keras.utils import np_utils
    uniques,ids=np.unique(arr,return_inverse=True)
    return np_utils.to_categorical(ids)

# Converts to categorical
y = np_to_categorical(change)

# Capture dataset
X = prices_today

# Split into training and test sets
from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    X, y, train_size=0.7, random_state=0)

# Build model and test
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

# Build model
model = Sequential()
model.add(Dense(16, input_shape=(6,)))
model.add(Activation('tanh'))
model.add(Dense(2))
model.add(Activation('softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd')

# Train
model.fit(train_X, train_y, verbose=1, batch_size=16,
  nb_epoch=5,show_accuracy=True)

# Test
loss, accuracy = model.evaluate(test_X, test_y, show_accuracy=True, verbose=0)

print("Test fraction correct (Accuracy) = {:.2f}".format(accuracy))
```


