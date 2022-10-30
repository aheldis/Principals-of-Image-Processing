# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

img = cv2.imread('Pink.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

# first approach
box_filter = np.ones((3, 3)) / 9
start_time = time.time()
blurred_img = cv2.filter2D(img, -1, box_filter)
end_time = time.time()
print("first approach's execution time:", end_time - start_time, "seconds")

# removing excessive rows and columns
blurred_img = blurred_img[1:-1, 1:-1]
blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res07.jpg", blurred_img)

# second approach
blurred_img = np.zeros(img.shape, dtype='uint8')
start_time = time.time()
for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):
        # convolution
        blurred_img[i][j][0] = np.sum(img[i - 1:i + 2, j - 1:j + 2, 0]) // 9
        blurred_img[i][j][1] = np.sum(img[i - 1:i + 2, j - 1:j + 2, 1]) // 9
        blurred_img[i][j][2] = np.sum(img[i - 1:i + 2, j - 1:j + 2, 2]) // 9
blurred_img = blurred_img[1:-1, 1:-1]
end_time = time.time()
print("second approach's execution time:", end_time - start_time, "seconds")
blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res08.jpg", blurred_img)

# third approach
start_time = time.time()
box_filter = [[1 / 9 for i in range(3)] for j in range(3)]

# calculate multiplication of each cell of box filter with image crops
for i in range(3):
    for j in range(3):
        box_filter[i][j] = box_filter[i][j] * img[i:img.shape[0] + i - 2, j:img.shape[1] + j - 2]
blurred_img = np.zeros((img.shape[0] - 2, img.shape[1] - 2, 3))

# calculate sum of box filter cells
for i in range(3):
    for j in range(3):
        blurred_img += box_filter[i][j]
blurred_img = np.array(blurred_img, dtype=np.uint8)
end_time = time.time()
print("third approach's execution time:", end_time - start_time, "seconds")
blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res09.jpg", blurred_img)
