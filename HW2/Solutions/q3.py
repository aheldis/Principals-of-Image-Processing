#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('books.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img);

# calculating coordinates of transformed patch image
def get_transformed_coordinates(coordinates):
    h = np.sqrt(np.sum((coordinates[1] - coordinates[0])**2))
    w = np.sqrt(np.sum((coordinates[2] - coordinates[0])**2))
    return np.array([[0, 0], [h, 0], [0, w], [h, w]])

yellow_coordinates = np.array([[210, 665], [105, 383], [394, 601], [288, 318]])
white_coordinates = np.array([[739, 356], [465, 403], [708, 153], [427, 204]])
black_coordinates = np.array([[969, 811], [668, 622], [1099, 610], [796, 421]])
books_coordinates = [yellow_coordinates, white_coordinates, black_coordinates]

# calculating transformation matrices
transforms = []
i = 0
for coordinates in books_coordinates:
    h, status = cv2.findHomography(coordinates, get_transformed_coordinates(coordinates))
    transforms.append(h)
    i += 1
    print("book number " + str(i) + " transformation matrix:\n", h)
    
# bilinear warping
def warp(src, des_shape, inv_t):
    des = np.zeros(des_shape)
    for i in range(des_shape[0]):
        for j in range(des_shape[1]):
            xp_yp = np.array([[i], [j], [1]], dtype=np.float32)
            x_y = np.dot(inv_t, xp_yp)
            x_y /= x_y[2]
            x, y = x_y[0][0], x_y[1][0]
            a = x - int(x)
            b = y - int(y)
            ma, mb = int(x), int(y)
            intensity = (1 - a) * (1 - b) * src[ma][mb]
            intensity += a * (1 - b) * src[ma + 1][mb]
            intensity += (1 - a) * b * src[ma][mb + 1]
            intensity += a * b * src[ma + 1][mb + 1]
            des[i][j] = intensity
    des = np.array(des, dtype=np.uint8)
    return des
    

r, g, b = cv2.split(img)
for i in range(3):
    inv_transform = np.linalg.inv(transforms[i])
    shape = np.array(get_transformed_coordinates(books_coordinates[i])[-1], dtype=np.int)
    des_r = warp(r, shape, inv_transform)
    des_g = warp(g, shape, inv_transform)
    des_b = warp(b, shape, inv_transform)
    des = cv2.merge((des_r, des_g, des_b))
    plt.imshow(des);
    plt.show()
    save = cv2.cvtColor(des, cv2.COLOR_RGB2BGR)
    cv2.imwrite("res" + str(16 + i) + ".jpg", save)

