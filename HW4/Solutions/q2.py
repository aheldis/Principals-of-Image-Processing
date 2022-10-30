# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')
window_size = 40
number_of_iterations = 3

img = cv2.imread('park.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize image
scale = 0.15
dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# extract features
data_features = np.zeros((resized_img.shape[0], resized_img.shape[1], 5), dtype=np.uint32)
data_features[:, :, :3] = cv2.cvtColor(resized_img, cv2.COLOR_RGB2HSV)
data_features[:, :, 3] = np.arange(resized_img.shape[0]).reshape((resized_img.shape[0], 1)) * 0.3
data_features[:, :, 4] = np.arange(resized_img.shape[1]).reshape((1, resized_img.shape[1])) * 0.3

# perform mean shift
centroids = np.array(data_features.reshape(-1, 5))
for k in range(10):  # this can be replaced with while True
    new_centroids = []
    for i in range(data_features.shape[0]):
        for j in range(data_features.shape[1]):
            # pick neighbours
            x_init, x_end = max(0, i - window_size // 2), min(i + window_size // 2 + 1, resized_img.shape[0])
            y_init, y_end = max(0, j - window_size // 2), min(j + window_size // 2 + 1, resized_img.shape[1])
            neighbours = data_features[x_init:x_end, y_init:y_end]
            neighbours = neighbours.reshape((-1, 5))
            centroid = centroids[i * data_features.shape[1] + j]
            neighbours = neighbours[np.sqrt(np.sum((neighbours - centroid) ** 2, axis=1)) < window_size]
            if len(neighbours) == 0:
                neighbours = [centroids[i]]
            # centroid equals to average of neighbours
            new_centroids.append(np.average(neighbours, axis=0))
    new_centroids = np.array(new_centroids)
    last_centroids = np.array(centroids)
    centroids = new_centroids

    print("end of iteration", k)

    # check converge
    if np.sum(centroids - last_centroids) == 0:
        break

# output the final image
new_img = np.array(centroids[:, :3], dtype=np.uint8).reshape(resized_img.shape)
new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)

# denoise the image using median blur
for c in range(3):
    new_img[:, :, c] = cv2.medianBlur(new_img[:, :, c], 3)

dim = (int(img.shape[1]), int(img.shape[0]))
resized_new_img = cv2.resize(new_img, dim, interpolation=cv2.INTER_AREA)

save = cv2.cvtColor(resized_new_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res05.jpg", save)
