# import libraries and read the points file
import numpy as np
import cv2
import matplotlib.pyplot as plt


# calculate distance between two given points
# distance is euclidean distance if points are not in polar space;
# otherwise, difference of the radius of given points is calculated
def distance(point1, point2, is_polar=False):
    if is_polar:
        # radius subtract
        return np.abs(point1[0] - point2[0])
    # euclidean distance
    return np.sqrt(np.sum((point1 - point2) ** 2))


# get two random initial centroids
def random_assignment(x, k, is_polar=False):
    indices = np.random.choice(len(x), k, replace=False)
    centroids = x[indices]
    distances = []
    for i in range(len(x)):
        distances.append([])
        for j in range(len(centroids)):
            distances[-1].append(distance(x[i], centroids[j], is_polar))
    distances = np.array(distances)
    points = np.array([np.argmin(i) for i in distances])
    return points


# perform k-means algorithm
def k_means(x, points, k, threshold=10 ** (-20), is_polar=False):
    last = np.array([[np.inf, np.inf], [np.inf, np.inf]])
    while True:
        # recalculate centroids with respect to point classes
        centroids = []
        for j in range(k):
            centroids.append(x[points == j].mean(axis=0))
        centroids = np.array(centroids)

        if np.sum(centroids - last) < threshold:
            break

        # calculate class of each point
        distances = []
        for j in range(len(x)):
            distances.append([])
            for k in range(len(centroids)):
                distances[-1].append(distance(x[j], centroids[k], is_polar))
        distances = np.array(distances)
        points = np.array([np.argmin(i) for i in distances])
        last = np.copy(centroids)
    return points


# read points file
with open('Points.txt') as points_file:
    n = int(points_file.readline())
    lines = points_file.readlines()

X = []
for line in lines:
    x, y = map(float, line.split())
    X.append([x, y])
X = np.array(X)

# plot the points in 2D
plt.plot(X[:, 0], X[:, 1], 'c.')
plt.savefig('res01.jpg')

# do twice
for i in range(2):
    points = random_assignment(X, k=2)
    preds = k_means(X, points, k=2)
    plt.plot(X[preds == 0][:, 0], X[preds == 0][:, 1], 'r.')
    plt.plot(X[preds == 1][:, 0], X[preds == 1][:, 1], 'g.')
    plt.savefig('res0' + str(i + 2) + '.jpg')

# make it polar!
mean = np.mean(X, axis=0)
rs = np.sqrt(np.sum((X - mean) ** 2, axis=1))
thetas = np.arctan((X[:, 1] - mean[1]) / (X[:, 0] - mean[0]))
polar_points = np.array([rs, thetas]).transpose()

points = random_assignment(polar_points, k=2, is_polar=True)
preds = k_means(polar_points, points, k=2, is_polar=True)
plt.plot(X[preds == 0][:, 0], X[preds == 0][:, 1], 'r.')
plt.plot(X[preds == 1][:, 0], X[preds == 1][:, 1], 'g.')
plt.savefig('res04.jpg')
