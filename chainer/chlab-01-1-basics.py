#!/usr/bin/env python
# Lab 1-1 Basics

import numpy as np
import chainer
"""Chainer is an open source deep learning framework written purely in Python on top of Numpy and CuPy Python libraries Chainer is notable for its early adoption of "define-by-run" scheme, as well as its performance on large scale systems."""

# Create Variable object.
a = chainer.Variable(np.array([1], dtype=np.float32))
b = chainer.Variable(np.array([2], dtype=np.float32))

# Variable object has basic arithmetic operators.
y = a * b

# Now y is a Variable object, with attribute "data".

print("{} should equal [ 2.]".format(y.data))

a = chainer.Variable(np.array([3], dtype=np.float32))
b = chainer.Variable(np.array([3], dtype=np.float32))

y = a * b
print("{} should equal [ 9.]".format(y.data))

"""
Expected output.
---
[ 2.] should equal [ 2.]
[ 9.] should equal [ 9.]
"""
