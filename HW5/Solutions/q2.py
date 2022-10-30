# import libraries and read images
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

deer = cv2.imread('res05.jpg')
deer = cv2.cvtColor(deer, cv2.COLOR_BGR2RGB)
plt.imshow(deer)
plt.show();

mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((10,10), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=3)
plt.imshow(mask, cmap='gray')
plt.show();

jungle = cv2.imread('res06.jpg')
jungle = cv2.cvtColor(jungle, cv2.COLOR_BGR2RGB)
plt.imshow(jungle)
plt.show();

# blur image using gaussian filter
def blur(img):
    sigma = 5
    shape = 2 * 3 *  sigma + 1
    gauss_filter_x = np.array([np.exp(-(i - shape//2)**2 / (2 * sigma**2)) for i in range(shape)])
    gauss_filter_x = gauss_filter_x / sum(gauss_filter_x)
    gauss_filter_x = gauss_filter_x.reshape((1, shape))
    gauss_filter_y = gauss_filter_x.reshape((shape, 1))
    blurred_img = cv2.filter2D(img, -1, gauss_filter_x)
    blurred_img = cv2.filter2D(blurred_img, -1, gauss_filter_y)
    return blurred_img

blurred_mask = blur(mask)
plt.imshow(blurred_mask, cmap='gray')
cv2.imwrite('blurred_mask.jpg', blurred_mask)
plt.show();

# a function to resize the image
def resize(img, scale):
    dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

resized_deer = resize(deer, 0.1)
plt.imshow(resized_deer)
plt.show()

resized_mask = resize(mask, 0.1)
plt.imshow(resized_mask, cmap='gray')
plt.show()

resized_blurred_mask = resize(blurred_mask, 0.1)
plt.imshow(resized_blurred_mask, cmap='gray')
plt.show()

# paste the source image in a image with size of the target
source_img = np.zeros(jungle.shape, dtype=np.uint8)
source_img[350:350+resized_deer.shape[0], 300:300+resized_deer.shape[1]] = resized_deer
plt.imshow(source_img)
plt.show()

source_mask = np.zeros((jungle.shape[0],jungle.shape[1]), dtype=np.uint8)
source_mask[350:350+resized_mask.shape[0], 300:300+resized_mask.shape[1]] = resized_mask
plt.imshow(source_mask, cmap='gray')
plt.show()

source_blurred_mask = np.zeros((jungle.shape[0],jungle.shape[1]), dtype=np.uint8)
source_blurred_mask[350:350+resized_blurred_mask.shape[0], 300:300+resized_blurred_mask.shape[1]] = resized_blurred_mask
plt.imshow(source_blurred_mask, cmap='gray')
plt.show()

# get laplacian of the source image
def laplacian(img):
    img = np.array(img, dtype=np.float64)
    _filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered_img = cv2.filter2D(img, -1, _filter)
    return filtered_img

right_side = laplacian(source_img)

# get the indices inside the mask
idx_arr = np.zeros(source_mask.shape, dtype=np.int32)
maximum = source_mask.shape[0] * source_mask.shape[1]
num = 0
for i in range(source_mask.shape[0]):
    for j in range(source_mask.shape[1]):
        if source_mask[i][j] == 0:
            idx_arr[i][j] = num
            num += 1
        else:
            idx_arr[i][j] = maximum

# get the left side of the equtions and solve them
left_side = []
right_side1 = []
right_side2 = []
right_side3 = []
row = []
col = []
for i in range(jungle.shape[0]):
    for j in range(jungle.shape[1]):
        if idx_arr[i][j] != maximum and max(idx_arr[i - 1][j], idx_arr[i + 1][j], idx_arr[i][j - 1],
                                            idx_arr[i][j + 1]) != maximum:
            left_side.append(-1)
            left_side.append(-1)
            left_side.append(4)
            left_side.append(-1)
            left_side.append(-1)
            right_side1.append(right_side[i][j][0])
            right_side2.append(right_side[i][j][1])
            right_side3.append(right_side[i][j][2])
            for _ in range(5):
                row.append(idx_arr[i, j])
            col.append(idx_arr[i - 1, j])
            col.append(idx_arr[i + 1, j])
            col.append(idx_arr[i, j])
            col.append(idx_arr[i, j - 1])
            col.append(idx_arr[i, j + 1])
        elif idx_arr[i][j] != maximum:
            left_side.append(1)
            right_side1.append(jungle[i][j][0])
            right_side2.append(jungle[i][j][1])
            right_side3.append(jungle[i][j][2])
            row.append(idx_arr[i][j])
            col.append(idx_arr[i][j])


n = len(np.where(idx_arr != maximum)[0])
left_side = csr_matrix((left_side, (row, col)), shape=(n, n))

x = []

x.append(spsolve(left_side, right_side1))
x.append(spsolve(left_side, right_side2))
x.append(spsolve(left_side, right_side3))

# get the final result
result = np.zeros(jungle.shape)
for c in range(3):
    for i in range(source_mask.shape[0]):
        for j in range(source_mask.shape[1]):
            if source_mask[i][j] > 0:
                result[i][j][c] = source_blurred_mask[i][j] / 255 * x[c][idx_arr[i][j]]
                result[i][j][c] += (1 - source_blurred_mask[i][j] / 255) * jungle[i][j][c]
            else:
                result[i][j][c] = jungle[i][j][c]

result = np.where(result > 255, 255, result)
result = np.where(result < 0, 0, result)
result = result.astype(np.uint8)
plt.imshow(result)
plt.show()

save = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("res07.jpg", save)