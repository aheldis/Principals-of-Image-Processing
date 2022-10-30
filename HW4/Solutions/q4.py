# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour

np.warnings.filterwarnings('ignore')

img = cv2.imread('birds.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize image
scale = 0.2
dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# convert RGB to gray
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

# sharpening the image
gray_img_sharpened = 2 * gray_img - cv2.blur(gray_img, (3, 3))

snakes = []
copies = []


# checking for mouse clicks
def click_event(event, x, y, flags, params):
    global snakes, resized_img, copies

    # compute segmentation for area aroung the clicked point
    if event == cv2.EVENT_LBUTTONDOWN:

        # localising the circle's center at x, y
        x_circle = x + 10 * np.cos(np.linspace(0, 2 * np.pi, 400))
        y_circle = y + 10 * np.sin(np.linspace(0, 2 * np.pi, 400))

        # generating a circle based on x1, x2
        snake = np.array([x_circle, y_circle]).T

        # computing the Active Contour for the given image
        snake = active_contour(gray_img_sharpened, snake, alpha=0.1, beta=0.1, gamma=0.8)
        snakes.append(snake)
        copies.append(np.copy(resized_img))
        for i in range(snake.shape[0] - 1):
            x1, y1 = snake[i]
            x2, y2 = snake[i + 1]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(resized_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.imshow('Click on the birds', cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

    # for undo :D
    if event == cv2.EVENT_RBUTTONDOWN:
        resized_img = copies[-1]
        copies = copies[:-1]
        snakes = snakes[:-1]
        cv2.imshow('Click on the birds', cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))


cv2.imshow('Click on the birds', cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
cv2.setMouseCallback('Click on the birds', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply on the base image
for snake in snakes:
    for i in range(snake.shape[0] - 1):
        x1, y1 = snake[i]
        x2, y2 = snake[i + 1]
        x1, y1, x2, y2 = int(np.round(x1 / scale)), int(np.round(y1 / scale)), int(np.round(x2 / scale)), int(
            np.round(y2 / scale))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=15)
save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res10.jpg", save)
