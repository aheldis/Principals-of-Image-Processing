# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# blur image using gaussian filter
def blur(img):
    shape, sigma = 20, 10
    gauss_filter_x = np.array([np.exp(-(i - shape // 2) ** 2 / (2 * sigma ** 2)) for i in range(shape)])
    gauss_filter_x = gauss_filter_x / sum(gauss_filter_x)
    gauss_filter_x = gauss_filter_x.reshape((1, shape))
    gauss_filter_y = gauss_filter_x.reshape((shape, 1))
    blurred_img = cv2.filter2D(img, -1, gauss_filter_x)
    blurred_img = cv2.filter2D(blurred_img, -1, gauss_filter_y)
    return blurred_img


img = cv2.imread('Flowers.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# convert pink color to yellow
h, s, v = cv2.split(img)
ht = np.where(h > 135, h - 120, h)
yellow_img = cv2.merge((ht, s, v))
rgb_img = cv2.cvtColor(yellow_img, cv2.COLOR_HSV2RGB);

r, g, b = cv2.split(rgb_img)
rt, gt, bt = blur(r), blur(g), blur(b)
blurred_img = cv2.merge((rt, gt, bt))

# blur and changing color together
rtt = np.where(h > 135, r, rt)
gtt = np.where(h > 135, g, gt)
btt = np.where(h > 135, b, bt)
result = cv2.merge((rtt, gtt, btt))

# save image
img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("res06.jpg", img)
