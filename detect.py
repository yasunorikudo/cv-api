#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import json
import requests
from skimage import io
from StringIO import StringIO


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    win = cv2.namedWindow('Object Detection', cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame,
                        (frame.shape[1] // 2, frame.shape[0] // 2))

        s = StringIO()
        io.imsave(s, img, plugin='pil')
        s.seek(0)

        files = {'file': s,}
        r = requests.post('http://192.168.12.34:8080/object', files=files)

        if r.text == '':
            continue
        results = json.loads(r.text)

        for ret in results:
            label = ret['label']
            x1, y1, x2, y2 = ret['bbox']

            # draw label
            t_size, b = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - t_size[1] - b),
                         (x1 + t_size[0], y2), (255, 0, 0), -1)
            cv2.putText(img, label, (x1, y2 - b / 2), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 1)

        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        cv2.imshow(win, img)

        key = cv2.waitKey(1)
        if key == 27:
            break

