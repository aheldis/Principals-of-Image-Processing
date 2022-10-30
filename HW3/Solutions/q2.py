# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# select a random patch for the beginning of texture synthesis
def random_patch(texture, margin=20, patch_size=50):
    rand_i = np.random.randint(texture.shape[0] - patch_size - 2 * margin)
    rand_j = np.random.randint(texture.shape[1] - patch_size - 2 * margin)
    return texture[rand_i + margin:rand_i + patch_size + margin, rand_j + margin:rand_j + patch_size + margin]


# template match using cv2.matchTemplate
def template_match(image, template, patch_size=50, margin=20):
    image_without_margin = image[margin:image.shape[0] - margin - patch_size][
                           margin:image.shape[1] - margin - patch_size]
    result = cv2.matchTemplate(image_without_margin, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc[0] + margin, max_loc[1] + margin


# find min_cut for patches filled from left to right or up to down
def min_cut(patch1, patch2, overlap, direction):
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


# a function to perform texture synthesis
def texture_synthesis(image, patch_size=280, margin=20, final_texture_size=2700, overlap_c=0.4):
    overlap = int(overlap_c * patch_size)
    n = int(np.ceil((final_texture_size - overlap) / (patch_size - overlap)))
    approx_size = (n * patch_size) - (n - 1) * overlap
    final_texture = np.zeros((approx_size, approx_size, 3), dtype=np.uint8)

    for i_total in range(n):
        for j_total in range(n):
            y = i_total * (patch_size - overlap)
            x = j_total * (patch_size - overlap)

            overlapping_patch = final_texture[y:y + patch_size, x:x + patch_size]
            patch_x, patch_y = template_match(image, overlapping_patch, margin=margin, patch_size=patch_size)
            new_patch = image[patch_y:patch_y + min(patch_size, overlapping_patch.shape[0]),
                        patch_x:patch_x + min(patch_size, overlapping_patch.shape[1])]

            mixed_patch = min_cut(overlapping_patch, new_patch, overlap, 1)
            mixed_patch = min_cut(overlapping_patch, mixed_patch, overlap, 0)

            # fill blank area with template matched patch
            r, g, b = cv2.split(mixed_patch)
            rn, gn, bn = cv2.split(new_patch)
            mixed_patch = cv2.merge((np.where(r + g + b == 0, rn, r),
                                     np.where(r + g + b == 0, gn, g),
                                     np.where(r + g + b == 0, bn, b)))
            final_texture[y:y + mixed_patch.shape[0], x:x + mixed_patch.shape[1]] = mixed_patch
    return final_texture[-2500:, -2500:]


# create all textures
file_names = ['texture02.png', 'texture11.jpeg', 'selected_texture1.png', 'selected_texture2.jpg']
patch_sizes = [280, 180, 200, 350]
overlap_cs = [0.4, 0.4, 0.4, 0.2]

for i in range(4):
    img = cv2.imread(file_names[i])
    result = texture_synthesis(img, patch_sizes[i], overlap_c=overlap_cs[i])
    show_side_by_side = np.full((3500, 5000, 3), 255, dtype=np.uint8)
    show_side_by_side[500:3000, 500:3000] = result
    st_y, en_y = 3500//2 - int(np.floor(img.shape[0]/2)), 3500//2 + int(np.ceil(img.shape[0]/2))
    st_x, en_x = 3500 + 1500//2 - int(np.floor(img.shape[1]/2)), 3500 + 1500//2 + int(np.ceil(img.shape[1]/2))
    show_side_by_side[st_y:en_y, st_x:en_x] = img
    cv2.imwrite("res" + str(11 + i) + ".jpg", show_side_by_side)
