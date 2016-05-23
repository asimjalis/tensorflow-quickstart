# Setting Up TensorFlow and Keras on Local Machine

## Overview

This document walks you through how to setup TensorFlow on your local
machine. 

## Conda

Follow these steps to install the *full* Conda (Anaconda)

<http://conda.pydata.org/docs/install/full.html>

<https://www.continuum.io/downloads>

**Make sure you get Python 2.7 *not* Python 3.5.**

## Install TensorFlow + Keras 

Install `pip` using `easy_install`:

    sudo easy_install pip

Install TensorFlow using `pip`:

    sudo easy_install --upgrade six
    sudo pip install --upgrade pip
    sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl

Install Keras:

    sudo pip install keras

Install additional packages:

    sudo pip install ipython
    sudo pip install jupyter

Start Python:

    ipython

Try string operations in TensorFlow: 

    import tensorflow as tf
    print tf.__version__
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

You should see this output:

    Hello, TensorFlow!

Try numeric calculations:

    a = tf.constant(10)
    b = tf.constant(32)
    print(sess.run(a + b))

You should see this output:

    42

