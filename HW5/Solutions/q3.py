# import libraries and read images
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('res08.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show();

img2 = cv2.imread('res09.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show();

# blur image using gaussian filter
def blur(img):
    img = np.float64(img)
    sigma = 3
    shape = 2 * 3 *  sigma + 1
    gauss_filter_x = np.array([np.exp(-(i - shape//2)**2 / (2 * sigma**2)) for i in range(shape)])
    gauss_filter_x = gauss_filter_x / sum(gauss_filter_x)
    gauss_filter_x = gauss_filter_x.reshape((1, shape))
    gauss_filter_y = gauss_filter_x.reshape((shape, 1))
    blurred_img = cv2.filter2D(img, -1, gauss_filter_x)
    blurred_img = cv2.filter2D(blurred_img, -1, gauss_filter_y)
    return blurred_img


# calculate the laplacian stack
pre1 = img1.astype(np.float64)
pre2 = img2.astype(np.float64)
laplaces1 = []
laplaces2 = []
for i in range(10):
    blur1 = blur(pre1)
    laplaces1.append(pre1 - blur1)
    blur2 = blur(pre2)
    laplaces2.append(pre2 - blur2)
    pre1, pre2 = blur1, blur2
    print('level ' + str(i) + ':')
    plt.imshow(laplaces1[-1].astype(np.uint8))
    plt.show()
    plt.imshow(laplaces2[-1].astype(np.uint8))
    plt.show()

laplaces1.append(blur1)
laplaces2.append(blur2)
plt.imshow(laplaces1[-1].astype(np.uint8))
plt.show()
plt.imshow(laplaces2[-1].astype(np.uint8))
plt.show()

# reverse the laplacian order
laplaces1 = laplaces1[::-1]
laplaces2 = laplaces2[::-1]

# blur image using gaussian filter
def blur_x(img, sigma):
    shape = 2 * 3 *  sigma + 1
    gauss_filter_x = np.array([np.exp(-(i - shape//2)**2 / (2 * sigma**2)) for i in range(shape)])
    gauss_filter_x = gauss_filter_x / sum(gauss_filter_x)
    gauss_filter_x = gauss_filter_x.reshape((1, shape))
    blurred_img = cv2.filter2D(img, -1, gauss_filter_x)
    return blurred_img

# get the final result using feathering
merged = np.zeros(img1.shape, dtype=np.float64)
sigma = int(1.5 ** 13)

for i in range(len(laplaces1)):
    mask = np.zeros(img1.shape, dtype=np.float64)
    mask[:, :mask.shape[1]//2] = 0
    mask[:, mask.shape[1]//2:] = 255
    mask = blur_x(mask, int(sigma))
    print('level ' + str(i) + ' with sigma=' + str(int(sigma)) + ':')
    merged += laplaces1[i] * (mask/255) + laplaces2[i] * (1 - mask/255)
    sigma = sigma / 1.1
    merged = np.where(merged>255, 255, merged)
    merged = np.where(merged<0, 0, merged)
    plt.imshow(merged.astype(np.uint8))
    plt.show()

save = cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("res10.jpg", save)