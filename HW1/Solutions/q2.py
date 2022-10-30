# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np


# equalize hist function
def equalize_hist(img):
    # calculate histogram
    hist = np.zeros(256)
    flatten = img.flatten()
    for intensity in flatten:
        hist[intensity] += 1

    # normalize hist
    normalized_hist = hist/sum(hist)

    # cummulative sum
    cumsum_hist = np.cumsum(normalized_hist)

    # create lookup table for new intensities
    new_intensities = np.floor(255 * cumsum_hist)

    # creating equalized hist image
    shape = img.shape
    img = img.flatten()
    new_img = np.array([new_intensities[item] for item in img], dtype=np.uint8)
    new_img = new_img.reshape(shape)

    return new_img


img = cv2.imread('Enhance2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# conditional log transformation
alpha = 0.2
beta = 0.03
r, g, b = cv2.split(img)
rt = np.where((r + g + b) / 3 < 40, np.uint8(255 / np.log(1 + 255 * alpha) * np.log(1 + alpha * r)),
              np.uint8(255 / np.log(1 + 255 * beta) * np.log(1 + beta * r)))
gt = np.where((r + g + b) / 3 < 40, np.uint8(255 / np.log(1 + 255 * alpha) * np.log(1 + alpha * g)),
              np.uint8(255 / np.log(1 + 255 * beta) * np.log(1 + beta * g)))
bt = np.where((r + g + b) / 3 < 40, np.uint8(255 / np.log(1 + 255 * alpha) * np.log(1 + alpha * b)),
              np.uint8(255 / np.log(1 + 255 * beta) * np.log(1 + beta * b)))
log_transformed = cv2.merge((rt, gt, bt))

rtt, gtt, btt = equalize_hist(rt), equalize_hist(gt), equalize_hist(bt)
img = cv2.merge((rtt, gtt, btt))

# save image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res02.jpg", img)
