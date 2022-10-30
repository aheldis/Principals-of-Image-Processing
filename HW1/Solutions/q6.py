# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram(img):
    hist = np.zeros(256)
    flatten = img.flatten()
    for intensity in flatten:
        hist[intensity] += 1
    return hist


def find_closest(value, array):
    # this function looks for closest value in array
    # if value exists in array; that value is the closest one
    if value in array:
        return value

    # find nearest value from the right
    save = value
    while value <= 255 and value not in array:
        value += 1
    increase_val = value
    if value not in array:  # in case value = 255 is not in array
        increase_val = np.inf

    # find nearest value from the left
    value = save
    while value >= 0 and value not in array:
        value -= 1
    decrease_val = value
    if value not in array:  # in case value = 0 is not in array
        decrease_val = np.inf
    value = save

    # find the nearest one
    _, value = min((abs(decrease_val - value), decrease_val), (abs(increase_val - value), increase_val))
    return value


def match_hist(pink, dark):
    # calculate equalized histogram of pink image
    hist = histogram(pink)
    normalized_hist = hist / np.sum(hist)
    pink_hist = np.floor(np.cumsum(normalized_hist) * 255)
    inverse_pink_hist = dict()
    for i in range(len(pink_hist)):
        inverse_pink_hist[pink_hist[i]] = i

    # calculate equalized histogram of dark image
    hist = histogram(dark)
    normalized_hist = hist / np.sum(hist)
    dark_hist = np.floor(np.cumsum(normalized_hist) * 255)

    # histogram matching
    flatten = dark.flatten()
    new_img = np.zeros(flatten.shape)
    for i, intensity in enumerate(flatten):
        value = find_closest(dark_hist[intensity], inverse_pink_hist)
        new_img[i] = inverse_pink_hist[value]
    new_img = np.array(new_img, dtype=np.uint8)
    new_img = new_img.reshape(dark.shape)

    return new_img


dark_img = cv2.imread('Dark.jpg')
dark_img = cv2.cvtColor(dark_img, cv2.COLOR_BGR2RGB)

pink_img = cv2.imread('Pink.jpg')
pink_img = cv2.cvtColor(pink_img, cv2.COLOR_BGR2RGB)

rp, gp, bp = cv2.split(pink_img)
rd, gd, bd = cv2.split(dark_img)
rt, gt, bt = match_hist(rp, rd), match_hist(gp, gd), match_hist(bp, bd)
img = cv2.merge((rt, gt, bt))

# histograms
plt.hist(x=np.arange(256), weights=histogram(rt), alpha=0.5, bins=256, color='r')
plt.hist(x=np.arange(256), weights=histogram(gt), alpha=0.5, bins=256, color='g')
plt.hist(x=np.arange(256), weights=histogram(bt), alpha=0.5, bins=256, color='b')
plt.title('histogram of matched Dark')
plt.savefig('res10.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("res11.jpg", img)
