#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('flowers.blur.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img);

############### a ################
# gaussian filter
f = img
sigma = 3
shape = 2 * 3 * sigma + 1
gauss_filter_x = np.array([np.exp(-(i - shape//2)**2 / (2 * sigma**2)) for i in range(shape)])
gauss_filter_x = gauss_filter_x /  sum(gauss_filter_x)
gauss_filter_x = gauss_filter_x.reshape((1, shape))
gauss_filter = np.dot(gauss_filter_x.transpose(), gauss_filter_x)
save = np.array(gauss_filter / np.max(gauss_filter) * 255, dtype=np.uint8)
plt.imshow(save, cmap='gray');
cv2.imwrite("res01.jpg", save)

# blurred image
blurred_img = cv2.filter2D(img, -1, gauss_filter)
plt.imshow(blurred_img);
save = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res02.jpg", save)

# unsharped mask
unsharped_mask = np.zeros(img.shape, dtype=np.float64)
unsharped_mask += img
unsharped_mask -= blurred_img
save = unsharped_mask - np.min(unsharped_mask)
save = save / np.max(save) * 255
save = np.array(save, dtype=np.uint8)
plt.imshow(save);
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite("res03.jpg", save)

# sharped image
alpha = 1.5
sharped_img = np.zeros(img.shape, dtype=np.float64)
sharped_img += img
sharped_img += alpha * unsharped_mask
sharped_img = np.where(sharped_img > 255.0, 255.0, sharped_img)
sharped_img = np.where(sharped_img < 0.0, 0.0, sharped_img)
sharped_img = np.array(sharped_img, dtype=np.uint8)
plt.imshow(sharped_img);
save = cv2.cvtColor(sharped_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res04.jpg", save)


############### b ################
# lapcian of gaussian filter
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_gauss = cv2.filter2D(gauss_filter, -1, laplacian)
save = laplacian_gauss - np.min(laplacian_gauss)
save = save / np.max(save) * 255
save = np.array(save, dtype=np.uint8)
plt.imshow(save, cmap='gray');
cv2.imwrite("res05.jpg", save)

# unsharped mask
unsharped_mask = np.array(img, dtype=np.float64)
unsharped_mask = cv2.filter2D(unsharped_mask, -1, laplacian_gauss)
save = unsharped_mask - np.min(unsharped_mask)
save = save / np.max(save) * 255
save = np.array(save, dtype=np.uint8)
plt.imshow(save);
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite("res06.jpg", save)

# sharped image
k = 10
sharped_img = np.zeros(img.shape, dtype=np.float64)
sharped_img += img
sharped_img -= k * unsharped_mask
sharped_img = np.where(sharped_img > 255.0, 255.0, sharped_img)
sharped_img = np.where(sharped_img < 0.0, 0.0, sharped_img)
sharped_img = np.array(sharped_img, dtype=np.uint8)
plt.imshow(sharped_img);
save = cv2.cvtColor(sharped_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res07.jpg", save)


############### c ################
# log magnitude of image
def get_magnitude(img):
    fft = np.fft.fft2(img)
    shifted_fft = np.fft.fftshift(fft)
    magnitude = np.abs(shifted_fft)
    return magnitude
r, g, b = cv2.split(img)
rm, gm, bm = get_magnitude(r), get_magnitude(g), get_magnitude(b)
magnitude = cv2.merge((rm, gm, bm))
magnitude = np.log(magnitude)
save = magnitude - np.min(magnitude)
save = save / np.max(magnitude) * 255
save = np.array(save, dtype=np.uint8)
plt.imshow(save);
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite("res08.jpg", save)

# gaussian high pass filter
D_0 = 150
gaussian_high_pass_filter = 1 - np.array([[np.exp(-((u - magnitude.shape[1]//2)**2 + (v - magnitude.shape[0]//2)**2) / (2 * D_0**2)) for u in range(magnitude.shape[1])] for v in range(magnitude.shape[0])])
save = np.array(gaussian_high_pass_filter / np.max(gaussian_high_pass_filter) * 255, dtype=np.uint8)
plt.imshow(save, cmap='gray');
cv2.imwrite("res09.jpg", save)

# sharped image in fourier
def get_shifted_fft(img):
    fft = np.fft.fft2(img)
    shifted_fft = np.fft.fftshift(fft)
    return shifted_fft

k = 1
r, g, b = cv2.split(img)
rm, gm, bm = get_shifted_fft(r), get_shifted_fft(g), get_shifted_fft(b)
_filter = 1 + k * gaussian_high_pass_filter
rmf, gmf, bmf = rm * _filter, gm * _filter, bm * _filter
filtered_magnitude = cv2.merge((np.abs(rmf), np.abs(gmf), np.abs(bmf)))
log_filtered_magnitude = np.log(filtered_magnitude)
save = log_filtered_magnitude - np.min(log_filtered_magnitude)
save = save / np.max(save) * 255
save = np.array(save, dtype=np.uint8)
plt.imshow(save);
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite("res10.jpg", save)

# sharped image
rf, gf, bf = np.fft.ifftshift(rmf), np.fft.ifftshift(gmf), np.fft.ifftshift(bmf)
r, g, b = np.real(np.fft.ifft2(rf)), np.real(np.fft.ifft2(gf)), np.real(np.fft.ifft2(bf))
sharped_img = cv2.merge((r, g, b))
sharped_img = np.where(sharped_img > 255.0, 255.0, sharped_img)
sharped_img = np.where(sharped_img < 0.0, 0.0, sharped_img)
sharped_img = np.array(sharped_img, dtype=np.uint8)
plt.imshow(sharped_img);
save = cv2.cvtColor(sharped_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res11.jpg", save)


############### d ################
# laplacian of gauss
laplacian_gauss = np.array([[4 * (np.pi ** 2) * ((u - magnitude.shape[1]//2)**2 + (v - magnitude.shape[0]//2)**2) for u in range(magnitude.shape[1])] for v in range(magnitude.shape[0])])
rmf, gmf, bmf = rm * laplacian_gauss, gm * laplacian_gauss, bm * laplacian_gauss
save = cv2.merge((np.abs(rmf), np.abs(gmf), np.abs(bmf)))
save = save - np.min(save)
save = save / np.max(save) * 255
save = np.array(save, dtype=np.uint8)
plt.imshow(save);
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite("res12.jpg", save)

# unsharped mask
rf, gf, bf = np.fft.ifftshift(rmf), np.fft.ifftshift(gmf), np.fft.ifftshift(bmf)
r, g, b = np.real(np.fft.ifft2(rf)), np.real(np.fft.ifft2(gf)), np.real(np.fft.ifft2(bf))
unsharped_mask = cv2.merge((r, g, b))
unsharped_mask = np.where(unsharped_mask > 255.0, 255.0, unsharped_mask)
unsharped_mask = np.where(unsharped_mask < 0.0, 0.0, unsharped_mask)
save = unsharped_mask - np.min(unsharped_mask)
save = np.array(save / np.max(save) * 255, dtype=np.uint8)
plt.imshow(save);
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite("res13.jpg", save)

# sharped image
k = 0.05
sharped_img = np.zeros(img.shape, dtype=np.float64)
sharped_img += img
sharped_img += k * unsharped_mask
sharped_img = np.where(sharped_img > 255.0, 255.0, sharped_img)
sharped_img = np.where(sharped_img < 0.0, 0.0, sharped_img)
sharped_img = np.array(sharped_img, dtype=np.uint8)
plt.imshow(sharped_img);
save = cv2.cvtColor(sharped_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res14.jpg", save)

