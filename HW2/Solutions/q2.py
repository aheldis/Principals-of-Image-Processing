#!/usr/bin/env python
# coding: utf-8

# In[33]:


# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')
img = cv2.imread('Greek-ship.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img);
plt.show()

template = cv2.imread('patch.png') 
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
plt.imshow(template);

# make black and white images
colored_img = img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray');
plt.show()
colored_template = template
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
plt.imshow(template, cmap='gray');
plt.show()

# preprocess image
def preprocess(image):
    image = cv2.GaussianBlur(image, (11, 11), 0)
    _filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    image = cv2.filter2D(image, -1, _filter)
    return image

img, template = preprocess(img), preprocess(template)

# calculate normalized cross correlation for a given template and index of image
def normalized_cross_correlation(template, image, m, n, diff_template, ssdt):
    k, l = template.shape[0], template.shape[1]
    patch_img = image[m:m+k, n:n+l]
    diff_patch_img = patch_img - np.mean(patch_img)
    ssdp = np.sqrt(np.sum(diff_patch_img ** 2)) # sum of squared diff_patch_img
    return np.sum(diff_template * diff_patch_img) / (ssdp * ssdt)

# calculate full ncc for whole image
def full_NCC(image, template, range1, range2):
    # calculating parameters
    image = np.array(image, dtype=np.float64)
    template = np.array(template, dtype=np.float64)
    diff_template = template - np.mean(template) # subtract mean from template is diff_template
    ssdt = np.sqrt(np.sum(diff_template**2)) # sqrt of sum of squared diff_template

    # calculate ncc matrix
    ncc_arr = np.zeros(image.shape, dtype=np.float64)
    for i in range(range1[0], range1[1]):
        for j in range(range2[0], range2[1]):
            ncc_arr[i][j] = normalized_cross_correlation(template, image, i, j, diff_template, ssdt)
    return ncc_arr

# make a pyramid of the image
def pyramid(image, template, depth=5):
    x, y = 0, 0
    for i in range(depth, -1, -1):
        scale = (1/2)**i
        dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        dim = (int(template.shape[1] * scale), int(template.shape[0] * scale))
        resized_tmp = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
        range1 = (x-1, x+1+1)
        range2 = (y-1, y+1+1)
        if i == depth:
            range1 = (0, resized_img.shape[0] - resized_tmp.shape[0])
            range2 = (0, resized_img.shape[1] - resized_tmp.shape[1])
        ncc_arr = full_NCC(resized_img, resized_tmp, range1, range2)
        max_idx = np.unravel_index(np.nanargmax(ncc_arr), ncc_arr.shape)
        x, y = max_idx[0] * 2, max_idx[1] * 2
    return max_idx

# find match
times = 0
while times < 7:
    idx = pyramid(img, template)
    shape = template.shape
    img[int(idx[0]):int(idx[0] + shape[0]), int(idx[1]):int(idx[1] + shape[1])] = 0
    colored_img = cv2.rectangle(colored_img, (idx[1], idx[0]), (idx[1] + template.shape[1], idx[0] + template.shape[0]), color=(255, 0, 0), thickness=5)
    times += 1
plt.imshow(colored_img)
plt.show()
save = cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res15.jpg", save)

