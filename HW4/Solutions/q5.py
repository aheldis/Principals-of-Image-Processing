# import libraries and read image
import numpy as np
import cv2

np.warnings.filterwarnings('ignore')

img = cv2.imread('tasbih.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize image
scale = 0.6
dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


# function to calculate euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# calculate points for initial snake
points = [np.array([-1, -1])]
start = False


def mouse_event(event, x, y, flags, param):
    global points, start
    if event == cv2.EVENT_LBUTTONDOWN:
        start = not start
    if start and event == cv2.EVENT_MOUSEMOVE:
        if euclidean_distance(points[-1], np.array([y, x])) >= 4:
            points.append(np.array([y, x]))
            cv2.circle(resized_img, (x, y), 2, (255, 255, 255), thickness=-1)
            cv2.imshow("Mark border", cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))


cv2.imshow("Mark border", cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
cv2.setMouseCallback('Mark border', mouse_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

points = points[1:]
points = np.array(points)


# get gradient of the image
def gradient(img):
    filter_x = np.array([[-1, 0, 1]])
    filter_y = filter_x.T
    dx = cv2.filter2D(img, -1, filter_x)
    dy = cv2.filter2D(img, -1, filter_y)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    return grad


# convert RGB to gray
gray_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2GRAY)

# denoise the image
grad = np.array(gradient(gray_img), dtype=np.float64)
denoised_img = np.array(grad / np.max(grad) * 255, dtype=np.uint8)
denoised_img = cv2.GaussianBlur(denoised_img, (21, 21), 0)
denoised_img = cv2.medianBlur(denoised_img, 13, 0)
denoised_img = cv2.medianBlur(denoised_img, 13, 0)
denoised_img[denoised_img < np.max(denoised_img) / 1.6] = 0
grad = denoised_img


# function to calculate targets for the second expression of the energy function
def get_targets(points, grad):
    # initialize with neighbours' average and total average
    n = len(points)
    targets = []
    mean_points = np.mean(points, axis=0)
    for i in range(len(points)):
        p1 = points[i]
        min_dist = np.inf
        for p2 in points:
            d = euclidean_distance(p1, p2)
            if d < min_dist and d > 0:
                min_dist = d
        mean_neighbours = (points[(i + 1) % n] + points[(i - 1) % n] + p1) / 3
        targets.append((mean_points - p1) + ((mean_neighbours - p1) ** 3) * 70 + p1)

    # indices of the points that are on the object
    indices = []
    for i in range(n):
        if grad[points[i][0], points[i][1]] > 0:
            indices.append(i)

    # update targets based on bisectors
    if len(indices) <= 1:
        return targets

    n = len(indices)
    for i in range(n):
        p1 = points[indices[i]]
        p2 = points[indices[(i + 1) % n]]
        bisector = np.array([p1[1] - p2[1], p2[0] - p1[0]])
        bisector_length = euclidean_distance(bisector, np.array([0, 0]))
        if bisector_length >= 0.01 and np.abs(indices[i] - indices[(i + 1) % n]) >= 2:
            target = (p1 + p2) / 2 - bisector / bisector_length * 200
            for j in range(indices[i] + 1, indices[(i + 1) % n]):
                targets[j] = target
    return targets


# calculate the energy
def energy(point1, point2, point3, grad, alpha=2, beta=100000, gamma=7000):
    first = alpha * (euclidean_distance(point1, point2)) ** 4
    second = beta * euclidean_distance(point2, point3) ** 2
    third = -gamma * grad[point2[0], point2[1]] ** 4
    return first + second + third


# for each snake iteration
def iterate_snakes(points, grad):
    # viterbi
    targets = get_targets(points, grad)
    n = len(points)

    # initialize
    neighbours = np.zeros((len(points), 9, 2), dtype=np.int32)
    for i in range(n):
        for neigh_i in range(0, 3):
            for neigh_j in range(0, 3):
                neighbours[i][neigh_i * 3 + neigh_j] = np.array(
                    [points[i][0] + neigh_i - 1, points[i][1] + neigh_j - 1])

    energy_arr = np.zeros((len(points), 9), dtype=np.float32)
    energy_idx = np.ones((len(points), 9), dtype=np.int32) * np.inf

    # dynamic programming
    for i in range(n + 1):
        for j in range(9):
            min_energy = np.inf
            for k in range(9):
                p1 = neighbours[(i + 1) % n][j]
                p2 = neighbours[i % n][k]
                if energy(p1, p2, targets[i % n], grad) + energy_arr[(i - 1) % n][k] < min_energy:
                    energy_arr[i % n][j] = energy(p1, p2, targets[i % n], grad) + energy_arr[(i - 1) % n][k]
                    energy_idx[i % n][j] = k
                    min_energy = energy_arr[i % n][j]
    # last point
    total_min_energy = np.inf
    total_min_idx = np.inf
    for i in range(9):
        if energy_arr[n - 1][i] < total_min_energy:
            total_min_energy = energy_arr[n - 1][i]
            total_min_idx = i

    # moving the points
    for i in range(n):
        points[i] = neighbours[i][int(energy_idx[i][total_min_idx])]
    return points


# perform active contours
save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("im0.jpg", save)

for iteration in range(100):
    last = np.copy(points)
    points = iterate_snakes(points, grad)
    points = np.array(points)
    saved_img = np.array(img)

    # draw snake
    for i in range(points.shape[0]):
        y1, x1 = points[i]
        y2, x2 = points[(i+1) % points.shape[0]]
        x1, y1, x2, y2 = int(np.round(x1/scale)), int(np.round(y1/scale)), int(np.round(x2/scale)), int(np.round(y2/scale))
        cv2.line(saved_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=5)

    save = cv2.cvtColor(saved_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("im" + str(iteration + 1) + ".jpg", save)
    cv2.imwrite("res11.jpg", save)
