#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import json
from multiprocessing import Process, Queue
import numpy as np
import requests
from skimage import io
from StringIO import StringIO


def encord(frame, q):
    img = frame[::2, ::-2].copy()
    # img = img[:, ::-1] # uncomment if you want flip the image
    s = StringIO()
    io.imsave(s, img[:, :, [2, 1, 0]], plugin='pil')
    s.seek(0)
    files = {'file': s,}
    q.put([img, files])


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    print('Press Esc to leave.')

    _, frame = cap.read()
    q = Queue()
    p = Process(target=encord, args=(frame, q))
    p.start()

    while True:
        # Get raw and encorded image.
        img, files = q.get()
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 2, h * 2))

        # Encord cap image with another process.
        _, frame = cap.read()
        q = Queue()
        p = Process(target=encord, args=(frame, q))
        p.start()

        # Send request to server.
        r = requests.post('http://192.168.12.34:8080/object', files=files)
        if r.text == '':
            continue
        objs = json.loads(r.text)

        for obj in objs:
            label = obj['label']
            x1, y1, x2, y2 = np.asarray(obj['bbox']) * 2

            # Draw label.
            t_size, b = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - t_size[1] - b),
                          (x1 + t_size[0], y2), (255, 0, 0), -1)
            cv2.putText(img, label, (x1, y2 - b / 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

        cv2.namedWindow('Detection', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Detection', img)

        key = cv2.waitKey(1)
        if key == 27:
            p.terminate()
            cap.release()
            cv2.destroyAllWindows()
            break
