from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

import os

model = load_model('models/unet.h5')
IMG_PATH = ('imgs/160.jpg')

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
img_ori = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 16))
#plt.imsave('Sonuc/sonuc1.png', img_ori)

IMG_WIDTH, IMG_HEIGHT = 256, 256


def preprocess(img):
    im = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / IMG_HEIGHT
        new_width = int(img.shape[1] / scale)
        diff = (IMG_WIDTH - new_width) // 2
        img = cv2.resize(img, (new_width, IMG_HEIGHT))
        # data setteki fotolar 256 yüksekliğinde eğitildiği için girilen datayı da 256ya çeviriyoruz yüksekliğini.
         # ona uygun bir genişlik ile de genişliği küçültüyoruz ki data setteki resimlerle uyumlu olsun

        im[:, diff:diff + new_width, :] = img


    else:
        scale = img.shape[1] / IMG_WIDTH
        new_height = int(img.shape[0] / scale)
        diff = (IMG_HEIGHT - new_height) // 2
        img = cv2.resize(img, (IMG_WIDTH, new_height))

        im[diff:diff + new_height, :, :] = img
    return im


img = preprocess(img)

plt.figure(figsize=(8, 8))
#plt.imsave('Sonuc/sonuc2.png', img)



input_img = img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3)).astype(np.float32) / 255.


pred = model.predict(input_img)

THRESHOLD = 0.5
EROSION = 1


def postprocess(img_ori, pred):
    h, w = img_ori.shape[:2]

    mask_ori = (pred.squeeze()[:, :, 1] > THRESHOLD).astype(np.uint8)
    max_size = max(h, w)
    result_mask = cv2.resize(mask_ori, dsize=(max_size, max_size))

    if h >= w:
        diff = (max_size - w) // 2
        if diff > 0:
            result_mask = result_mask[:, diff:-diff]
    else:
        diff = (max_size - h) // 2
        if diff > 0:
            result_mask = result_mask[diff:-diff, :]

    result_mask = cv2.resize(result_mask, dsize=(w, h))


    result_mask *= 255

    result_mask = cv2.GaussianBlur(result_mask, ksize=(9, 9), sigmaX=5, sigmaY=5)

    return result_mask


mask = postprocess(img_ori, pred)

plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plt.imshow(pred[0, :, :, 1])
plt.subplot(1, 2, 2)
#plt.imsave('Sonuc/sonuc3.png', mask)

converted_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
plt.imsave('Sonuc/160.png', converted_mask)

result_img = cv2.subtract(converted_mask, img_ori)
plt.figure(figsize=(16, 16))
#plt.imsave('Sonuc/sonuc5.png', result_img)
result_img = cv2.subtract(converted_mask, result_img)

plt.figure(figsize=(16, 16))
#plt.imsave('Sonuc/sonuc6.png', result_img)




