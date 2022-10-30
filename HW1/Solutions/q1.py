# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


img = cv2.imread('Enhance1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# log transformation
alpha = 0.5
log_transformed = np.uint8(255 / np.log(1 + 255 * alpha) * np.log(1 + alpha * img))

# equalize hist
r, g, b = cv2.split(log_transformed)
rt, gt, bt = equalize_hist(r), equalize_hist(g), equalize_hist(b)
img = cv2.merge((rt, gt, bt))

# save image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res01.jpg", img)
