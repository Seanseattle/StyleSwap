import cv2
import numpy as np
import paddle


def cv2paddle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = paddle.to_tensor(img, dtype='float32').unsqueeze(axis=0)
    img = (img / 255.0 - 0.5) / 0.5
    return img

def paddle2cv(img):
    img = img.numpy()[0]
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img