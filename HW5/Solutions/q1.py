# import libraries and read images
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('res01.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1);
plt.show()

img2 = cv2.imread('res02.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2);
plt.show()

# # checking for mouse clicks
# def click_event(event, x, y, flags, params):
#     global points, img, copies
#
#     # detect a click on the border
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append([y, x])
#         copies.append(np.copy(img))
#         cv2.circle(img, (x, y), 2, (100, 100, 100), thickness=-1)
#         cv2.putText(img, str(len(points)), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
#         cv2.imshow('Click on the border', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#
#     # for undo :D
#     if event == cv2.EVENT_RBUTTONDOWN:
#         img = copies[-1]
#         copies = copies[:-1]
#         points = points[:-1]
#         cv2.imshow('Click on the border', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#
#     # define variables for mouse event
# img = np.copy(img1)
# points = []
# copies = []
#
# # call the click_event function for selecting the points
# cv2.imshow('Click on the border', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# cv2.setMouseCallback('Click on the border', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # write the position of the points in file points1.txt
# points1 = list(points)
# with open('points1.txt', 'w') as points_file1:
#     for point in points1:
#         points_file1.write(str(point[0]) + ' ' + str(point[1]) + '\n')
#
# # save the image with points
# img = np.copy(img1)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# for i in range(len(points1)):
#     point = points1[i]
#     cv2.circle(img, (point[1], point[0]), 1, (200, 200, 0), thickness=-1)
#     cv2.putText(img, str(i + 1), (point[1] + 3, point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 0), 1, cv2.LINE_AA)
#
# cv2.imwrite("img1_with_points.jpg", img)
#
# # define variables for mouse event
# img = np.copy(img2)
# points = []
# copies = []
#
# # call the click_event function for selecting the points
# cv2.imshow('Click on the border', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# cv2.setMouseCallback('Click on the border', click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # write the position of the points in file points1.txt
# points2 = list(points)
# with open('points2.txt', 'w') as points_file2:
#     for point in points2:
#         points_file2.write(str(point[0]) + ' ' + str(point[1]) + '\n')
#
# # save the image with points
# img = np.copy(img2)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# for i in range(len(points2)):
#     point = points2[i]
#     cv2.circle(img, (point[1], point[0]), 1, (200, 200, 0), thickness=-1)
#     cv2.putText(img, str(i + 1), (point[1] + 3, point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 0), 1, cv2.LINE_AA)
#
# cv2.imwrite("img2_with_points.jpg", img)


# read the position of the points from files
points1 = []
with open('points1.txt', 'r') as points_file1:
    lines = points_file1.readlines()
    for line in lines:
        points1.append(list(map(int, line.split())))

points2 = []
with open('points2.txt', 'r') as points_file2:
    lines = points_file2.readlines()
    for line in lines:
        points2.append(list(map(int, line.split())))

img = cv2.imread('img1_with_points.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img);
plt.show()

img = cv2.imread('img2_with_points.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img);
plt.show()

# add corners to the points
points1.append([0, img.shape[1]//2])
points1.append([img.shape[0] - 1, img.shape[1]//2])
points1.append([img.shape[0]//3, 0])
points1.append([img.shape[0]//3, img.shape[1] - 1])
points1.append([0, 0])
points1.append([img.shape[0] - 1, 0])
points1.append([0, img.shape[1] - 1])
points1.append([img.shape[0] - 1, img.shape[1] - 1])

points2.append([0, img.shape[1]//2])
points2.append([img.shape[0] - 1, img.shape[1]//2])
points2.append([img.shape[0]//3, 0])
points2.append([img.shape[0]//3, img.shape[1] - 1])
points2.append([0, 0])
points2.append([img.shape[0] - 1, 0])
points2.append([0, img.shape[1] - 1])
points2.append([img.shape[0] - 1, img.shape[1] - 1])

# calculate the triangles of the first image
rect = (0, 0, img.shape[0], img.shape[1])
subdiv = cv2.Subdiv2D(rect);
for p in points1:
    subdiv.insert((p[0], p[1]))
triangles1 = subdiv.getTriangleList()

# show the triangles with line
img = np.copy(img1)
for t in triangles1:
    t = np.array(t, dtype=np.int64)
    cv2.line(img, (t[1], t[0]), (t[3], t[2]), (0, 200, 200), thickness=1)
    cv2.line(img, (t[1], t[0]), (t[5], t[4]), (0, 200, 200), thickness=1)
    cv2.line(img, (t[3], t[2]), (t[5], t[4]), (0, 200, 200), thickness=1)
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("img1_with_lines.jpg", img)

# arrange the triangle points
def arrange_triangle_points(points):
    return np.array([[points[0], points[1]],
                     [points[2], points[3]],
                     [points[4], points[5]]])

arranged_triangles1 = []
for i in range(len(triangles1)):
    arranged_triangles1.append(arrange_triangle_points(triangles1[i]))
triangles1 = np.array(arranged_triangles1, np.int32)

# get the triangles of the second image
points1, points2 = np.array(points1), np.array(points2)
triangles2 = []
for i in range(len(triangles1)):
    t = triangles1[i]
    idx0 = np.where(np.sum(points1 == t[0], axis=1) == 2)[0][0]
    idx1 = np.where(np.sum(points1 == t[1], axis=1) == 2)[0][0]
    idx2 = np.where(np.sum(points1 == t[2], axis=1) == 2)[0][0]
    triangles2.append(np.array([points2[idx0], points2[idx1], points2[idx2]]))
triangles2 = np.array(triangles2, dtype=np.int32)

# show the triangles with line
img = np.copy(img2)
for t in triangles2:
    cv2.line(img, (t[0][1], t[0][0]), (t[1][1], t[1][0]), (0, 200, 200), thickness=1)
    cv2.line(img, (t[0][1], t[0][0]), (t[2][1], t[2][0]), (0, 200, 200), thickness=1)
    cv2.line(img, (t[1][1], t[1][0]), (t[2][1], t[2][0]), (0, 200, 200), thickness=1)
plt.imshow(img)
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("img2_with_lines.jpg", img)

# get peripheral rectangle
def get_rectangle(triangle):
    min_y, max_y = np.min(triangle[:, 0]), np.max(triangle[:, 0])
    min_x, max_x = np.min(triangle[:, 1]), np.max(triangle[:, 1])
    return min_y, min_x, max_y - min_y + 1, max_x - min_x + 1


# warp the triangle of the source to destination with same size as the main image
def warp(img, triangle1, triangle2):
    img = np.array(img, dtype=np.uint8)
    rectangle1, rectangle2 = get_rectangle(triangle1), get_rectangle(triangle2)
    crop1, crop2 = [], []

    for _ in range(3):
        crop1.append(((triangle1[_][1] - rectangle1[1]), (triangle1[_][0] - rectangle1[0])))
        crop2.append(((triangle2[_][1] - rectangle2[1]), (triangle2[_][0] - rectangle2[0])))

    crop_img = img[rectangle1[0]:rectangle1[0]+rectangle1[2], rectangle1[1]:rectangle1[1]+rectangle1[3]]
    mask = np.zeros((rectangle2[2], rectangle2[3], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(crop2, dtype=np.int32), (1, 1, 1));

    new_img = np.zeros(img.shape, dtype=np.uint8)
    y0, y1, x0, x1 = rectangle2[0], rectangle2[0]+rectangle2[2], rectangle2[1], rectangle2[1]+rectangle2[3]
    transform = cv2.getAffineTransform(np.array(crop1, dtype=np.float32), np.array(crop2, dtype=np.float32))
    new_img[y0:y1, x0:x1] = cv2.warpAffine(crop_img, transform, (rectangle2[3], rectangle2[2]),
                                           None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) * mask
    return new_img


save = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.imwrite('im0.jpg', save)
cv2.imwrite('im90.jpg', save)


# get the weighted average of the triangle points,
# warp the source triangle to destination,
# and get the weighted average of the inside of the triangles in each frame
for i in range(1, 45):
    mid_img = np.zeros(img.shape, dtype=np.uint8)
    mid_triangles = ((1.0 - i / 45) * triangles1 + (i / 45) * triangles2).astype(np.int32)
    for j in range(len(triangles1)):
        warp_img1 = warp(img1, triangles1[j], mid_triangles[j])
        warp_img2 = warp(img2, triangles2[j], mid_triangles[j])
        add = np.round((1.0 - i / 45) * warp_img1 + (i / 45) * warp_img2).astype(np.uint8)
        mid_img = np.where(mid_img==0, add, mid_img)
    plt.imshow(mid_img)
    plt.show()
    save = cv2.cvtColor(mid_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('im' + str(i) + '.jpg', save)
    cv2.imwrite('im' + str(90 - i) + '.jpg', save)
    if i == 15:
        cv2.imwrite('res03.jpg', save)
    if i == 30:
        cv2.imwrite('res04.jpg', save)

save = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
cv2.imwrite('im45.jpg', save)