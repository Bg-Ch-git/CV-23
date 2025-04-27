import numpy as np
import numpy.fft as FFT


def align(img, g_coord):
    hight = img.shape[0]
    margin = 0
    if hight % 3 == 0:
        imgs = np.split(img, 3)
    elif hight % 3 == 1:
        imgs = np.split(img[:-1, :], 3)
        margin = 1
    else:
        imgs = np.split(img[:-2, :], 3)
        margin = 1

    size = imgs[0].shape
    width_m = size[1] // 10
    hight_m = size[0] // 10
    hight = size[0]

    for i in range(3):
        imgs[i] = imgs[i][hight_m:-hight_m, width_m:-width_m]

    img_r = imgs[2]
    img_g = imgs[1]
    img_b = imgs[0]

    ft_r = FFT.fft2(img_r)
    ft_g = FFT.fft2(img_g)
    ft_b = FFT.fft2(img_b)

    metr_r = FFT.ifft2(ft_g * np.conjugate(ft_r))
    metr_b = FFT.ifft2(ft_g * np.conjugate(ft_b))

    vec_r = np.unravel_index(np.argmax(metr_r, axis=None), metr_r.shape)
    vec_b = np.unravel_index(np.argmax(metr_b, axis=None), metr_b.shape)

    b_row = (g_coord[0] - margin - hight_m - hight -
             vec_b[0]) % img_b.shape[0] + margin + hight_m
    b_col = (g_coord[1] - width_m - vec_b[1]) % img_b.shape[1] + width_m

    r_row = (g_coord[0] - margin - hight_m - hight - vec_r[0]
             ) % img_r.shape[0] + margin + hight_m + 2 * hight
    r_col = (g_coord[1] - width_m - vec_r[1]) % img_r.shape[1] + width_m

    hight = img_b.shape[0]
    width = img_b.shape[1]
    h_b_f = 0
    h_b_b = 0
    h_g_b = 0
    h_g_f = 0
    h_r_b = 0
    h_r_f = 0
    if hight > 2 * vec_b[0]:
        if hight > 2 * vec_r[0]:
            if vec_b[0] > vec_r[0]:
                h_b_b = 0
                h_b_f = -vec_b[0] if vec_b[0] != 0 else hight
                h_g_b = vec_b[0]
                h_g_f = hight
                h_r_b = vec_b[0] - vec_r[0]
                h_r_f = -vec_r[0] if vec_r[0] != 0 else hight
            else:
                h_r_b = 0
                h_r_f = -vec_r[0] if vec_r[0] != 0 else hight
                h_g_b = vec_r[0]
                h_g_f = hight
                h_b_b = vec_r[0] - vec_b[0]
                h_b_f = -vec_b[0] if vec_b[0] != 0 else hight
        else:
            h_b_b = 0
            h_b_f = -vec_b[0]-(hight-vec_r[0]) if -vec_b[0] - \
                (hight-vec_r[0]) != 0 else hight
            h_g_b = vec_b[0]
            h_g_f = -(hight-vec_r[0]) if hight-vec_r[0] != 0 else hight
            h_r_b = vec_b[0]+(hight-vec_r[0])
            h_r_f = hight
    else:
        if hight > 2 * vec_r[0]:
            h_r_b = 0
            h_r_f = -vec_r[0]-(hight-vec_b[0]) if -vec_r[0] - \
                (hight-vec_b[0]) != 0 else hight
            h_g_b = vec_r[0]
            h_g_f = -(hight-vec_b[0]) if hight-vec_b[0] != 0 else hight
            h_b_b = vec_r[0]+(hight-vec_b[0])
            h_b_f = hight
        else:
            if vec_b[0] > vec_r[0]:
                h_b_b = hight - vec_b[0]
                h_b_f = -(vec_b[0] - vec_r[0]) if vec_b[0] - \
                    vec_r[0] != 0 else hight
                h_g_b = 0
                h_g_f = -(hight - vec_r[0]) if hight - vec_r[0] != 0 else hight
                h_r_b = hight - vec_r[0]
                h_r_f = hight
            else:
                h_r_b = hight - vec_r[0]
                h_r_f = -(vec_r[0] - vec_b[0]) if vec_b[0] - \
                    vec_r[0] != 0 else hight
                h_g_b = 0
                h_g_f = -(hight - vec_b[0]) if hight - vec_r[0] != 0 else hight
                h_b_b = hight - vec_b[0]
                h_b_f = hight

    w_b_f = 0
    w_b_b = 0
    w_g_b = 0
    w_g_f = 0
    w_r_b = 0
    w_r_f = 0
    if width > 2 * vec_b[1]:
        if width > 2 * vec_r[1]:
            if vec_b[1] > vec_r[1]:
                w_b_b = 0
                w_b_f = -vec_b[1] if vec_b[1] != 0 else width
                w_g_b = vec_b[1]
                w_g_f = width
                w_r_b = vec_b[1] - vec_r[1]
                w_r_f = -vec_r[1] if vec_r[1] != 0 else width
            else:
                w_r_b = 0
                w_r_f = -vec_r[1] if vec_r[1] != 0 else width
                w_g_b = vec_r[1]
                w_g_f = width
                w_b_b = vec_r[1] - vec_b[1]
                w_b_f = -vec_b[1] if vec_b[1] != 0 else width
        else:
            w_b_b = 0
            w_b_f = -vec_b[1]-(width-vec_r[1]) if -vec_b[1] - \
                (width-vec_r[1]) != 0 else width
            w_g_b = vec_b[1]
            w_g_f = -(width-vec_r[1]) if width-vec_r[1] != 0 else width
            w_r_b = vec_b[1]+(width-vec_r[1])
            w_r_f = width
    else:
        if width > 2 * vec_r[1]:
            w_r_b = 0
            w_r_f = -vec_r[1]-(width-vec_b[1]) if -vec_r[1] - \
                (width-vec_b[1]) != 0 else width
            w_g_b = vec_r[1]
            w_g_f = -(width-vec_b[1]) if width-vec_b[1] != 0 else width
            w_b_b = vec_r[1]+(width-vec_b[1])
            w_b_f = width
        else:
            if vec_b[1] > vec_r[1]:
                w_b_b = width - vec_b[1]
                w_b_f = -(vec_b[1] - vec_r[1]) if vec_b[1] - \
                    vec_r[1] != 0 else width
                w_g_b = 0
                w_g_f = -(width - vec_r[1]) if width - vec_r[1] != 0 else width
                w_r_b = width - vec_r[1]
                w_r_f = width
            else:
                w_r_b = width - vec_r[1]
                w_r_f = -(vec_r[1] - vec_b[1]) if vec_b[1] - \
                    vec_r[1] != 0 else width
                w_g_b = 0
                w_g_f = -(width - vec_b[1]) if width - vec_r[1] != 0 else width
                w_b_b = width - vec_b[1]
                w_b_f = width

    return np.dstack((img_r[h_r_b:h_r_f, w_r_b:w_r_f], img_g[h_g_b:h_g_f, w_g_b:w_g_f], img_b[h_b_b:h_b_f, w_b_b:w_b_f])), (b_row, b_col), (r_row, r_col)
