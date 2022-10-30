#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries and read image
import numpy as np
import cv2
import matplotlib.pyplot as plt

near = cv2.imread('res19-near.jpg') 
near = cv2.cvtColor(near, cv2.COLOR_BGR2RGB)
plt.imshow(near);
plt.show()

far = cv2.imread('res20-far.jpg') 
far = cv2.cvtColor(far, cv2.COLOR_BGR2RGB)
plt.imshow(far);

# calculate euclidean distance
def get_euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

# calculate transforms needed in order to align far image to near image
def transform(near, far):
    # translate to origin
    t = -far[0]
    t1 = np.array([[1, 0, t[1]], [0, 1, t[0]], [0, 0, 1]])
    
    # rotate
    d_near = near[1] - near[0]
    d_far = far[1] - far[0]
    theta = np.arctan((d_near[0]) / (d_near[1])) - np.arctan((d_far[0]) / (d_far[1]))
    t2 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    # scale
    dist_near = get_euclidean_distance(near[0], near[1])
    dist_far = get_euclidean_distance(far[0], far[1])
    a = dist_near / dist_far
    t3 = np.array([[a, 0, 0], [0, a, 0], [0, 0, 1]])
    
    # translate to near
    t = near[0]
    t4 = np.array([[1, 0, t[1]], [0, 1, t[0]], [0, 0, 1]])
    
    t = np.dot(np.dot(t4, t3), np.dot(t2, t1))
    return t
     
# warp
def warp(near, far, near_eyes, far_eyes):
    t = transform(near_eyes, far_eyes)
    far_shape_t = np.array(np.dot(t, np.array([[far.shape[0]], [far.shape[1]], [1]])), dtype=np.int)
    far_t = cv2.warpPerspective(far, t, (min(near.shape[1], far_shape_t[1][0]), min(near.shape[0], far_shape_t[0][0])))
    near_t = near[:min(near.shape[0], far_shape_t[0][0]), :min(near.shape[1], far_shape_t[1][0])]
    return far_t, near_t


far_eyes = np.array([[108, 302], [109, 334]])
near_eyes = np.array([[213, 261], [208, 338]])
far_t, near_t = warp(near, far, near_eyes, far_eyes)

plt.imshow(near_t);
plt.show()
save = cv2.cvtColor(near_t, cv2.COLOR_RGB2BGR)
cv2.imwrite("res21-near.jpg", save)

plt.imshow(far_t);
save = cv2.cvtColor(far_t, cv2.COLOR_RGB2BGR)
cv2.imwrite("res22-far.jpg", save)

# no comments needed!
def show_log_fft_one_channel(img):
    fft = np.fft.fft2(img)
    shift = np.fft.fftshift(fft)
    return shift, np.log(np.abs(shift))

# no comments needed!
def show_log_fft_three_channel(img, name):
    r, g, b = cv2.split(img)
    shift_r, r = show_log_fft_one_channel(r)
    shift_g, g = show_log_fft_one_channel(g)
    shift_b, b = show_log_fft_one_channel(b)
    log_fft = cv2.merge((r, g, b))
    show = log_fft - np.min(log_fft)
    show = np.array(show / np.max(show) * 255, dtype=np.uint8)
    plt.imshow(show);
    plt.show()
    save = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, save)
    return shift_r, shift_g, shift_b

near_fft = show_log_fft_three_channel(near_t, 'res23-dft-near.jpg')
far_fft = show_log_fft_three_channel(far_t, 'res24-dft-far.jpg')

# create gaussian low pass and high pass filters r=25, s=60
def get_filter(sigma, is_high, shape):
    D_0 = sigma
    gaussian_low_pass_filter = np.array([[np.exp(-((u - shape[1]//2)**2 + (v - shape[0]//2)**2) / (2 * D_0**2)) for u in range(shape[1])] for v in range(shape[0])])
    gaussian_high_pass_filter = 1 - gaussian_low_pass_filter
    if is_high:
        _filter = gaussian_high_pass_filter
    else:
        _filter = gaussian_low_pass_filter
    _filter = np.where(_filter < 0.5, 0, _filter)
    save = _filter - np.min(_filter)
    save = np.array(save / np.max(save) * 255, dtype=np.uint8)
    plt.imshow(save, cmap='gray');
    plt.show()
    if is_high:
        cv2.imwrite('res25-highpass-' + str(sigma) + '.jpg' , save)
    else:
        cv2.imwrite('res26-lowpass-' + str(sigma) + '.jpg' , save)
    return _filter

high_pass = get_filter(25, True, near_t.shape)
low_pass = get_filter(60, False, far_t.shape)

# filtering transformed images
def filter_image(img, f, name):
    r, g, b = img
    rt, gt, bt = r*f, g*f, b*f
    rts, gts, bts = np.abs(rt), np.abs(gt), np.abs(bt)
    rts = np.where(rts == 0, 0, np.log(rts))
    gts = np.where(gts == 0, 0, np.log(gts))
    bts = np.where(bts == 0, 0, np.log(bts))
    filtered_fft = cv2.merge((rts, gts, bts))
    save = filtered_fft - np.min(filtered_fft)
    save = np.array(save / np.max(save) * 255, dtype=np.uint8)
    plt.imshow(save)
    plt.show();
    save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, save)
    return rt, gt, bt
    
near_high_passed = filter_image(near_fft, high_pass, 'res27-highpassed.jpg')
far_low_passed = filter_image(far_fft, low_pass, 'res28-lowpassed.jpg')

# calculating radius of gaussian filter (nonzero values)
def get_gaussian_radius(image, is_high):
    if is_high:  
        _range = range(image.shape[1] // 2, 0, -1)
    else:
        _range = range(image.shape[1])
    for i in _range:
        if image[image.shape[0] // 2][i] > 0:
            return image.shape[1] // 2 - i

# merge filtered and transformed image using weighted average
def merge_filters(near, far):
    nr, ng, nb = near
    fr, fg, fb = far
    r1 = get_gaussian_radius(nr, True)
    r2 = get_gaussian_radius(fr, True)
    r1, r2 = min(r1, r2), max(r1, r2)
    shape = near[0].shape
    mfr, mfg, mfb = np.zeros(shape, dtype=np.complex), np.zeros(shape, dtype=np.complex), np.zeros(shape, dtype=np.complex)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # no overlap
            if (i - shape[0]//2)**2 + (j - shape[1]//2)**2 > max(r1, r2)**2:
                mfr[i][j] = nr[i][j]
                mfg[i][j] = ng[i][j]
                mfb[i][j] = nb[i][j]
            # overlapping part -> weighted average
            elif (i - shape[0]//2)**2 + (j - shape[1]//2)**2 > min(r1, r2)**2:
                dist = get_euclidean_distance(np.array([i, j]), np.array([shape[0] // 2, shape[1] // 2]))
                mfr[i][j] = (dist - r1) / (r2 - r1) * nr[i][j] + (dist - r2) / (r1 - r2) * fr[i][j]
                mfg[i][j] = (dist - r1) / (r2 - r1) * ng[i][j] + (dist - r2) / (r1 - r2) * fg[i][j]
                mfb[i][j] = (dist - r1) / (r2 - r1) * nb[i][j] + (dist - r2) / (r1 - r2) * fb[i][j]
            # no overlap
            else:
                mfr[i][j] = fr[i][j]
                mfg[i][j] = fg[i][j]
                mfb[i][j] = fb[i][j]
    r, g, b = np.abs(mfr), np.abs(mfg), np.abs(mfb)
    r = np.where(r == 0, 0, np.log(r))
    g = np.where(g == 0, 0, np.log(g))
    b = np.where(b == 0, 0, np.log(b))
    merged_fft = cv2.merge((r, g, b))
    save = merged_fft - np.min(merged_fft)
    save = np.array(save / np.max(save) * 255, dtype=np.uint8)
    plt.imshow(save)
    plt.show();
    save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
    cv2.imwrite('res29-hybrid.jpg', save)
    return mfr, mfg, mfb
merged_fft = merge_filters(near_high_passed, far_low_passed)

# calculate final image
def get_filtered_image(fft):
    r, g, b = fft
    rt, gt, bt = np.fft.ifftshift(r), np.fft.ifftshift(g), np.fft.ifftshift(b)
    rt, gt, bt = np.fft.ifft2(rt), np.fft.ifft2(gt), np.fft.ifft2(bt)
    rt, gt, bt = np.real(rt), np.real(gt), np.real(bt)
    filtered_img = cv2.merge((rt, gt, bt))
    filtered_img = np.where(filtered_img > 255.0, 255.0, filtered_img)
    filtered_img = np.where(filtered_img < 0.0, 0.0, filtered_img)
    filtered_img = np.array(filtered_img, dtype=np.uint8)
    plt.imshow(filtered_img)
    plt.show();
    save = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('res30-hybrid-near.jpg', save)
    return filtered_img
filtered_image = get_filtered_image(merged_fft)    

# scaled final image
shape = (filtered_image.shape[1] // 5, filtered_image.shape[0] // 5)
save = cv2.resize(filtered_image, shape, interpolation=cv2.INTER_AREA)
plt.imshow(save)
plt.show();
save = cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
cv2.imwrite('res31-hybrid-far.jpg', save)

