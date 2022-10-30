# import libraries
import numpy as np
import cv2


# initial clusters
def init(k, data_features):
    size = int(np.sqrt(img.shape[0] * img.shape[1] / k))
    clusters = []
    for i in range(size // 2, img.shape[0], size):
        for j in range(size // 2, img.shape[1], size):
            clusters.append(data_features[i][j])
    return np.array(clusters), size


# get gradient of the image
def gradient(img):
    img = np.array(img, dtype=np.float64)
    filter_x = np.array([[-1, 0, 1]])
    filter_y = filter_x.T
    dx = cv2.filter2D(img, -1, filter_x)
    dy = cv2.filter2D(img, -1, filter_y)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    return grad


# update clusters such that they have less gradient
def better_neighbour(grad, clusters, data_features):
    new_clusters = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        selected_grad = grad[cluster[3] - 2:cluster[3] + 3, cluster[4] - 2:cluster[4] + 3]
        idx = np.unravel_index(np.argmin(selected_grad, axis=None), selected_grad.shape)
        x, y = cluster[3] + idx[0] - 2, cluster[4] + idx[1] - 2
        new_cluster = data_features[x][y]
        new_clusters.append(new_cluster)
    return np.array(new_clusters)


# assign a cluster to each pixel
def assign_clusters_to_pixels(clusters, s, img, data_features):
    min_d = np.ones((img.shape[0], img.shape[1]), dtype=np.float64) * np.inf
    min_idx = np.ones((img.shape[0], img.shape[1]), dtype=np.uint32) - 2
    for cluster_num in range(len(clusters)):
        cluster = clusters[cluster_num]
        min_i, max_i = max(cluster[3] - s, 0), min(cluster[3] + s + 1, data_features.shape[0])
        min_j, max_j = max(cluster[4] - s, 0), min(cluster[4] + s + 1, data_features.shape[1])
        neighbours = data_features[min_i:max_i, min_j:max_j]
        d_lab = np.sqrt(np.sum((neighbours[:, :, :3] - cluster[:3]) ** 2, axis=2))
        d_xy = np.sqrt(np.sum((neighbours[:, :, 3:] - cluster[3:]) ** 2, axis=2))
        alpha = 75 / s
        d = d_lab + alpha * d_xy

        min_idx_block = min_idx[min_i:max_i, min_j:max_j]
        min_d_block = min_d[min_i:max_i, min_j:max_j]
        min_idx_block[d < min_d_block] = cluster_num
        min_d_block[d < min_d_block] = d[d < min_d_block]

    for cluster_num in range(len(clusters)):
        cluster = clusters[cluster_num]
        min_i, max_i = max(cluster[3] - s, 0), min(cluster[3] + s + 1, data_features.shape[0])
        min_j, max_j = max(cluster[4] - s, 0), min(cluster[4] + s + 1, data_features.shape[1])
        min_idx_block = min_idx[min_i:max_i, min_j:max_j]
        neighbours = data_features[min_i:max_i, min_j:max_j]
        clusters[cluster_num] = np.mean(neighbours[min_idx_block == cluster_num], axis=0)
    return min_idx, clusters


# to remove small noise clusters
def enforce_connectivity(clusters_idx, clusters, s, img):
    for cluster_num in range(len(clusters)):
        cluster = clusters[cluster_num]
        clusters_idx_block = clusters_idx[max(cluster[3] - s, 0):min(cluster[3] + s, img.shape[0]),
                             max(cluster[4] - s, 0):min(cluster[4] + s, img.shape[1])]
        array_for_detecting_connected_components = np.uint8(clusters_idx_block == cluster_num)
        connected_labels, connected_img = cv2.connectedComponents(array_for_detecting_connected_components)
        for label in range(connected_labels):
            if len(connected_img[connected_img == label]) < s ** 2 / 9:
                cluster_block = np.where(connected_img == label)
                if clusters_idx_block[cluster_block[0][0], cluster_block[1][0]] == cluster_num:
                    j = cluster_block[1][0] + max(cluster[4] - s, 0)
                    min_i = np.min(cluster_block[0]) + max(cluster[3] - s, 0)
                    clusters_idx_block[cluster_block] = clusters_idx[min_i - 1][j]
    return clusters_idx


# draw borders of clusters
def draw_borders(clusters, img, name):
    borders = gradient(clusters)

    r, g, b = cv2.split(img)
    r = np.where(borders != 0, 255, r)
    g = np.where(borders != 0, 255, g)
    b = np.where(borders != 0, 0, b)
    new_img = cv2.merge((r, g, b))

    save = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, save)


# perform slic
def slic(k, data_features, img, name, rounds=20):
    clusters, s = init(k, data_features)
    grad = gradient(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    clusters = better_neighbour(grad, clusters, data_features)
    for i in range(rounds):
        clusters_idx, clusters = assign_clusters_to_pixels(clusters, s, img, data_features)
    clusters_idx = enforce_connectivity(clusters_idx, clusters, s, img)
    draw_borders(clusters_idx, img, name)


# read image
img = cv2.imread('slic.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# extract features
data_features = np.zeros((img.shape[0], img.shape[1], 5), dtype=np.uint32)
data_features[:, :, :3] = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
data_features[:, :, 3] = np.arange(img.shape[0]).reshape((img.shape[0], 1))
data_features[:, :, 4] = np.arange(img.shape[1]).reshape((1, img.shape[1]))

ks = [64, 256, 1024, 2048]
names = ['res06.jpg', 'res07.jpg', 'res08.jpg', 'res09.jpg']
for i in range(4):
    slic(ks[i], data_features, img, names[i])
