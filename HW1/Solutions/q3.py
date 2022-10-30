# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff


# separate the channels
def separate(img):
    return img[:len(img) // 3], img[len(img) // 3:2 * (len(img) // 3)], img[2 * (len(img) // 3):3 * (len(img) // 3)]


def convert_to_uint8(r, g, b):
    img = cv2.merge((r, g, b))
    return np.array(img / 256, dtype=np.uint8)


def show_img(r, g, b):
    img = convert_to_uint8(r, g, b)
    plt.imshow(img)
    plt.show()


def sum_of_error(source, target):
    return np.sum(((source - smooth(source)) - (target - smooth(target))) ** 2)


def smooth(img):
    box_filter = np.ones((4, 4)) / 16
    return cv2.filter2D(img, -1, box_filter)


def shift(img, i, j):
    img = np.roll(img, i, axis=0)
    img = np.roll(img, j, axis=1)

    if i > 0:
        img[:i] = 0
    elif i < 0:
        img[img.shape[0] + i:] = 0
    if j > 0:
        img[:, :j] = 0
    elif j < 0:
        img[:, img.shape[1] + j:] = 0
    return img


def prokudin_gorskii(target, source, other_channel, source_color, scale=0.2):
    res_x_shift, res_y_shift = 0, 0
    past_y_shift, past_x_shift = 0, 0
    if np.min(target.shape) > 400:
        dim = (int(source.shape[1] * scale), int(source.shape[0] * scale))
        resized_target = cv2.resize(target, dim, interpolation=cv2.INTER_AREA)
        resized_source = cv2.resize(source, dim, interpolation=cv2.INTER_AREA)
        resized_other_channel = cv2.resize(other_channel, dim, interpolation=cv2.INTER_AREA)
        res_x_shift, res_y_shift, _ = prokudin_gorskii(resized_target, resized_source, resized_other_channel,
                                                       source_color, scale)
        res_x_shift, res_y_shift = int(res_x_shift / scale), int(res_y_shift / scale)
        source = shift(source, res_y_shift, res_x_shift)

    x_shift = 0
    y_shift = 0
    free = (1 / scale)
    min_error = sum_of_error(source, target)
    best_source = source
    for i_shift in range(-int(free), int(free + 1)):
        for j_shift in range(-int(free), int(free + 1)):
            source_shifted = shift(source, i_shift, j_shift)
            if sum_of_error(source_shifted, target) < min_error:
                min_error = sum_of_error(source_shifted, target)
                best_source = source_shifted
                y_shift = res_y_shift + i_shift
                x_shift = res_x_shift + j_shift

    source = best_source

    if source_color == 'g':
        show_img(target, source, other_channel)
    else:
        show_img(target, other_channel, source)
    return x_shift, y_shift, source


def register(img):
    b, g, r = separate(img)
    print("channel green:")
    x_shift, y_shift, g = prokudin_gorskii(r, g, b, 'g')
    print("Shift in direction of x:", x_shift, ", Shift in direction of y:", y_shift)
    print("channel blue:")
    x_shift, y_shift, b = prokudin_gorskii(r, b, g, 'b')
    print("Shift in direction of x:", x_shift, ", Shift in direction of y:", y_shift)
    return convert_to_uint8(r, g, b)


def derivative(img):
    filter_x = np.array([[-1, 0, 1]])
    filter_y = filter_x.T
    dx = cv2.filter2D(img, -1, filter_x)
    dy = cv2.filter2D(img, -1, filter_y)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    return grad


def get_margin(img):
    r, g, b = cv2.split(img)
    dv = derivative((r + g + b) / 3)
    last = 0
    le, ri, up, do = 0, 0, 0, 0
    for i, x in enumerate(dv[img.shape[1] // 2]):
        if (i < img.shape[1] / 20 or i > img.shape[1] * 19 / 20) and (x > 1):
            if last < img.shape[1] / 20:
                le, ri = last, img.shape[1]
            if i - last > 2 / 3 * img.shape[1]:
                le, ri = last, i
            last = i

    last = 0
    for i, x in enumerate(dv.T[img.shape[0] // 2]):
        if (i < img.shape[0] / 20 or i > img.shape[0] * 19 / 20) and (x > 1):
            if last < img.shape[0] / 20:
                up, do = last, img.shape[0]
            if i - last > 2 / 3 * img.shape[0]:
                up, do = last, i
            last = i
    return le, ri, up, do


amir = tiff.imread('amir.tif')  # master-pnp-prok-01800-01886a
print('Amir:')
amir_reg = register(amir)
l, r, u, d = get_margin(amir_reg)
cropped = amir_reg[u:d, l:r]
img = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
cv2.imwrite("res03-Amir.jpg", img)

print('Mosque:')
mosque = tiff.imread('mosque.tif')  # master-pnp-prok-01800-01833a
mosque_reg = register(mosque)
l, r, u, d = get_margin(mosque_reg)
cropped = mosque_reg[u:d, l:r]
img = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
cv2.imwrite("res04-Mosque.jpg", img)

print('Train:')
train = tiff.imread('train.tif')  # master-pnp-prok-00400-00458a
train_reg = register(train)
l, r, u, d = get_margin(train_reg)
cropped = train_reg[u:d, l:r]
img = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
cv2.imwrite("res05-Train.jpg", img)
