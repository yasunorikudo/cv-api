#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bottle import route, run, request
import json
import numpy as np
from skimage import io
from StringIO import StringIO

from ssd import Detector
from generate import Style


@route('/object', method='POST')
def object():
    try:
        image = request.files.get('file')
        image = io.imread(StringIO(image.file.read()))
        objs = detector(image.astype(np.float32) / 255.)
        return json.dumps(objs)

    except Exception as e:
        print str(type(e)), e


@route('/style', method='POST')
def style():
    try:
        image = request.files.get('file')
        image = io.imread(StringIO(image.file.read()))
        result_image = style(image)
        s = StringIO()
        np.save(s, result_image)
        return s.getvalue()

    except Exception as e:
        print str(type(e)), e


if __name__ == '__main__':
    detector = Detector()
    style = Style()
    run(host='0.0.0.0', port=8080, debug=True, reloader=True)
