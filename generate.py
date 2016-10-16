#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from chainer import cuda, Variable, serializers
from net import FastStyleNet


class Style(object):
    def __init__(self):
        self._model = FastStyleNet()
        serializers.load_npz('composition.model', self._model)
        cuda.get_device(0).use()
        self._model.to_gpu()

    def __call__(self, image):
        image = cuda.cupy.asarray(
            image, dtype=cuda.cupy.float32).transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)
        x = Variable(image)

        y = self._model(x)
        result_image = cuda.to_cpu(y.data)

        result_image = result_image.transpose(0, 2, 3, 1)
        result_image = result_image.reshape((result_image.shape[1:]))
        result_image = np.uint8(result_image)

        return result_image
