# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# a function to remove swimmer and birds from the images
def delete_areas(img, areas):
    for area in areas:
        img[area[0][0]:area[1][0], area[0][1]:area[1][1]] = 0
    return img


# template match using cv2.matchTemplate
def template_match(image, template, patch_size=50, margin=20):
    image_without_margin = image[margin:image.shape[0] - margin - patch_size][
                           margin:image.shape[1] - margin - patch_size]
    result = cv2.matchTemplate(image_without_margin, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc[0] + margin, max_loc[1] + margin


# find min_cut for patches filled from left to right or up to down
def min_cut_from_left_or_up(patch1, patch2, overlap, direction):
    errors = np.sum((patch1 - patch2) ** 2, axis=2)
    dp = np.ones(errors.shape) * np.inf
    best_neighbors = [[(0, 0) for j in range(errors.shape[1])] for i in range(errors.shape[0])]

    if direction == 0:
        dp[0, :] = errors[0, :]
        for i in range(1, dp.shape[0]):
            for j in range(dp.shape[1]):
                best_neighbors[i][j] = (i - 1, min(j, overlap))
                for neighbor in range(max(0, j - 1), min(j + 2, dp.shape[1], overlap)):
                    if dp[i - 1][neighbor] + errors[i][j] < dp[i][j]:
                        dp[i][j] = dp[i - 1][neighbor] + errors[i][j]
                        best_neighbors[i][j] = (i - 1, neighbor)
    else:
        dp[:, 0] = errors[:, 0]
        for j in range(1, dp.shape[1]):
            for i in range(dp.shape[0]):
                best_neighbors[i][j] = (min(i, overlap), j - 1)
                for neighbor in range(max(0, i - 1), min(i + 2, dp.shape[0], overlap)):
                    if dp[neighbor][j - 1] + errors[i][j] < dp[i][j]:
                        dp[i][j] = dp[neighbor][j - 1] + errors[i][j]
                        best_neighbors[i][j] = (neighbor, j - 1)

    if direction == 0:
        i, j = dp.shape[0] - 1, np.argmin(dp[-1])
    else:
        i, j = np.argmin(dp[:, -1]), dp.shape[1] - 1

    mixed_patch = np.array(patch2)
    while i > 0 or j > 0:
        neigh_i, neigh_j = best_neighbors[i][j]
        if direction == 0:
            mixed_patch[i, :j] = patch1[i, :j]
        else:
            mixed_patch[:i, j] = patch1[:i, j]
        i, j = neigh_i, neigh_j

    return mixed_patch


# find min_cut for different flips
def min_cut(patch1, patch2, overlap, flip):
    if flip is not None:
        patch1 = cv2.flip(np.array(patch1), flip)
        patch2 = cv2.flip(np.array(patch2), flip)
    patch2 = min_cut_from_left_or_up(patch1, patch2, overlap, 1)
    mixed_patch = min_cut_from_left_or_up(patch1, patch2, overlap, 0)
    if flip is not None:
        mixed_patch = cv2.flip(np.array(mixed_patch), flip)
    return mixed_patch


# a function to perform texture synthesis
def texture_synthesis(image, to_fill, patch_size=200, margin=0, overlap_c=0.3):
    overlap = int(overlap_c * patch_size)

    ni = int(np.ceil((to_fill.shape[0] - overlap) / (patch_size - overlap)))
    nj = int(np.ceil((to_fill.shape[1] - overlap) / (patch_size - overlap)))

    for i_total in range(ni):
        for j_total in range(nj):
            y = i_total * (patch_size - overlap)
            x = j_total * (patch_size - overlap)

            overlapping_patch = to_fill[y:y + patch_size, x:x + patch_size]
            patch_x, patch_y = template_match(image, overlapping_patch, margin=margin, patch_size=patch_size)
            new_patch = image[patch_y:patch_y + min(patch_size, overlapping_patch.shape[0]),
                        patch_x:patch_x + min(patch_size, overlapping_patch.shape[1])]

            mixed_patch = min_cut(overlapping_patch, new_patch, overlap, -1)  # flip both sides
            mixed_patch = min_cut(overlapping_patch, mixed_patch, overlap, 1)  # flip y axis
            mixed_patch = min_cut(overlapping_patch, mixed_patch, overlap, 0)  # flip x axis
            mixed_patch = min_cut(overlapping_patch, mixed_patch, overlap, 2)  # no flips

            # fill blank area with template matched patch
            r, g, b = cv2.split(mixed_patch)
            rn, gn, bn = cv2.split(new_patch)
            mixed_patch = cv2.merge((np.where(r + g + b == 0, rn, r),
                                     np.where(r + g + b == 0, gn, g),
                                     np.where(r + g + b == 0, bn, b)))
            to_fill[y:y + mixed_patch.shape[0], x:x + mixed_patch.shape[1]] = mixed_patch
    return to_fill


# a function to perform template match for each blanck area seperately
def fill_holes(img, areas, using_area, patch_size=200, overlap_c=0.3):
    img = np.array(img)
    for area in areas:
        y_start, y_end = area[0][0] - 50, area[1][0] + 50
        x_start, x_end = area[0][1] - 50, area[1][1] + 50
        to_fill = np.array(img[y_start:y_end, x_start:x_end])
        img[y_start:y_end, x_start:x_end] = texture_synthesis(using_area, to_fill, patch_size=200, overlap_c=0.3)
    return img


areas1 = [[[60, 310], [172, 542]], [[746, 802], [931, 976]], [[608, 1125], [725, 1257]]]
areas2 = [[[678, 718], [1173, 970]]]

birds = cv2.imread('im03.jpg')
birds_with_hole = delete_areas(birds, areas1)
new_birds = fill_holes(birds_with_hole, areas1, birds_with_hole[200:, :700], patch_size=100, overlap_c=0.3)
cv2.imwrite("res15.jpg", new_birds)

swimmer = cv2.imread('im04.jpg')
swimmer_with_hole = delete_areas(swimmer, areas2)
new_swimmer = fill_holes(swimmer_with_hole, areas2, swimmer_with_hole[1200:], patch_size=200, overlap_c=0.3)
cv2.imwrite("res16.jpg", new_swimmer)
