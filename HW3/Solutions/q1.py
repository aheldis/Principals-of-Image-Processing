# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# create hough space rho between (-diagonal//2, diagonal//2) and theta between 0 and pi
# rho steps = 400 and theta steps = 400
def get_hough_space(img, rho_size=400, theta_size=400):
    hough_space = np.zeros((rho_size, theta_size))
    diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                for theta_idx in range(theta_size):
                    theta = theta_idx / theta_size * np.pi
                    rho = (i - img.shape[0]//2) * np.cos(theta) + (j - img.shape[1]//2) * np.sin(theta)
                    rho_idx = int(rho / diagonal * rho_size + rho_size/2)
                    hough_space[rho_idx][theta_idx] += 1
    return hough_space


# a function to get the highest points in hough space and return rhos and thetas
def get_local_maximas(hough_space, number_of_lines=17):
    # get local maximas
    arr = np.array(hough_space)
    local_maximas = []
    for i in range(number_of_lines):
        y, x = np.unravel_index(arr.argmax(), hough_space.shape)
        arr[y-4:y+5, x-4:x+5] = 0
        local_maximas.append([y, x])

    # detect outliers
    dists = []
    for i in range(len(local_maximas)):
        i_dists = []
        for j in range(len(local_maximas)):
            item1 = local_maximas[i]
            item2 = local_maximas[j]
            if i != j:
                i_dists.append([int(np.sqrt((item1[0]-item2[0])**2 + (item1[1]-item2[1])**2)), i, j])
        dists.append(min(i_dists))
    dists = np.array(dists)

    Q1 = np.percentile(dists[:, 0], 25, interpolation = 'midpoint')
    Q3 = np.percentile(dists[:, 0], 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    # select lines that are not outlier
    selected_maximas = []
    for i in range(len(local_maximas)):
        if dists[i][0] < threshold:
            selected_maximas.append(local_maximas[i])
    return local_maximas, selected_maximas


# a function to draw the lines of chess board
def draw_lines(img, maximas, rho_size=400, theta_size=400):
    arr = maximas
    diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    for i_and_j in arr:
        rho_idx, theta_idx = i_and_j[0], i_and_j[1]
        rho = (rho_idx - rho_size/2) * 1.0 / rho_size * diagonal
        theta = theta_idx * 1.0 / theta_size * np.pi
        y = int(rho * np.cos(theta) + img.shape[0]/2)
        x = int(rho * np.sin(theta) + img.shape[1]/2)
        img = cv2.line(np.array(img),
                       (x - int(2000 * np.cos(theta)), y + int(2000 * np.sin(theta))),
                       (x + int(2000 * np.cos(theta)), y - int(2000 * np.sin(theta))),
                       (0, 0, 255), 5)
    return img


# calculate the determinant
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


# calculate intersection of two lines
def line_intersection(line1, line2):
    dx = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    dy = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    determinan = det(dx, dy)
    if determinan == 0:
        return None

    d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
    x = det(d, dx) // determinan
    y = det(d, dy) // determinan
    return x, y


# calculate two dots of a given line (line a given using theta_idx and rho_idx)
def get_line(img, rho_idx, theta_idx, rho_size, theta_size, diagonal):
    rho = (rho_idx - rho_size/2) * 1.0 / rho_size * diagonal
    theta = theta_idx * 1.0 / theta_size * np.pi
    y = int(rho * np.cos(theta) + img.shape[0]/2)
    x = int(rho * np.sin(theta) + img.shape[1]/2)
    x1 = x - int(50 * np.cos(theta))
    y1 = y + int(50 * np.sin(theta))
    x2 = x + int(50 * np.cos(theta))
    y2 = y - int(50 * np.sin(theta))
    return [[x1, y1], [x2, y2]]


# a function to draw the dots given rhos and thetas of the lines
def draw_dots(img, maximas, rho_size=400, theta_size=400):
    arr = maximas
    diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    dotted_img = np.array(img)
    img = np.array(img)
    dots = []
    for ij1 in arr:
        rho_idx, theta_idx = ij1[0], ij1[1]
        line1 = get_line(img, rho_idx, theta_idx, rho_size, theta_size, diagonal)
        for ij2 in arr:
            rho_idx, theta_idx = ij2[0], ij2[1]
            line2 = get_line(img, rho_idx, theta_idx, rho_size, theta_size, diagonal)
            intersection = line_intersection(line1, line2)
            if intersection is not None:
                dotted_img = cv2.circle(dotted_img, intersection, radius=10, color=(255, 0, 255), thickness=-1)
    return dotted_img


img1 = cv2.imread('im01.jpg')
edges1 = cv2.Canny(img1, 325, 350, L2gradient=True)
cv2.imwrite("res01.jpg", edges1)

img2 = cv2.imread('im02.jpg')
edges2 = cv2.Canny(img2, 325, 350, L2gradient=True)
cv2.imwrite("res02.jpg", edges2)


hough_space1 = get_hough_space(edges1)
save = np.uint8(hough_space1 / np.max(hough_space1) * 255)
cv2.imwrite("res03-hough-space.jpg", save)

hough_space2 = get_hough_space(edges2)
save = np.uint8(hough_space2 / np.max(hough_space2) * 255)
cv2.imwrite("res04-hough-space.jpg", save)


local_maximas1, selected_maximas1 = get_local_maximas(hough_space1)
drawn_img_all_lines1 = draw_lines(img1, local_maximas1)
cv2.imwrite("res05-lines.jpg", drawn_img_all_lines1)

local_maximas2, selected_maximas2 = get_local_maximas(hough_space2)
drawn_img_all_lines2 = draw_lines(img2, local_maximas2)
cv2.imwrite("res06-lines.jpg", drawn_img_all_lines2)


drawn_img_selected_lines1 = draw_lines(img1, selected_maximas1)
cv2.imwrite("res07-chess.jpg", drawn_img_selected_lines1)

drawn_img_selected_lines2 = draw_lines(img2, selected_maximas2)
cv2.imwrite("res08-chess.jpg", drawn_img_selected_lines2)


dotted_img1 = draw_dots(img1, selected_maximas1)
cv2.imwrite("res09-corners.jpg", dotted_img1)


dotted_img2 = draw_dots(img2, selected_maximas2)
cv2.imwrite("res10-corners.jpg", dotted_img2)
