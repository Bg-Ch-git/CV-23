import numpy as np


def get_bayer_masks(n_rows, n_cols):
    mask_red = np.array([[False, True],
                         [False, False]])
    mask_green = np.array([[True, False],
                           [False, True]])
    mask_blue = np.array([[False, False],
                          [True, False]])
    rows_n = n_rows // 2 if n_rows % 2 == 0 else n_rows // 2 + 1
    cols_n = n_cols // 2 if n_cols % 2 == 0 else n_cols // 2 + 1
    red = np.tile(mask_red, (rows_n, cols_n))[:n_rows, :n_cols]
    green = np.tile(mask_green, (rows_n, cols_n))[:n_rows, :n_cols]
    blue = np.tile(mask_blue, (rows_n, cols_n))[:n_rows, :n_cols]
    return np.dstack((red, green, blue))


def get_colored_img(raw_img):
    img = np.dstack((raw_img, raw_img, raw_img))
    mask = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    img[mask == False] = 0
    return img


def bilinear_interpolation(colored_img):
    n_rows = colored_img.shape[0]
    n_cols = colored_img.shape[1]
    mask = get_bayer_masks(n_rows, n_cols)
    count = np.array(mask)
    res_count = np.zeros(count.shape)
    res_count[0, :, :] = 1
    res_count[:, 0, :] = 1
    res_count[-1, :, :] = 1
    res_count[:, -1, :] = 1
    res_img = np.zeros(colored_img.shape)
    for i in range(3):
        for j in range(3):
            res_count[1:-1, 1:-1, :] += count[i:n_rows -
                                              (2-i), j:n_cols-(2-j), :]
            res_img[1:-1, 1:-1, :] += colored_img[i:n_rows -
                                                  (2-i), j:n_cols-(2-j), :]
    res = np.copy(colored_img)
    res[mask == False] = res_img[mask == False] / res_count[mask == False]
    return np.array(res, dtype='uint8')


def improved_interpolation(raw_img):
    n_rows = raw_img.shape[0]
    n_cols = raw_img.shape[1]
    raw_img = np.array(raw_img, dtype='int')
    res = np.zeros((n_rows, n_cols, 3))
    for i in range(2, n_rows - 2):
        for j in range(2, n_cols - 2):
            if i % 2 == 0:
                if j % 2 == 0:
                    res[i, j, 0] = (5*raw_img[i, j]-(raw_img[i-1, j-1]+raw_img[i-1, j+1]+raw_img[i+1, j-1]+raw_img[i+1, j+1] +
                                    raw_img[i, j-2]+raw_img[i, j+2])+0.5*(raw_img[i-2, j]+raw_img[i+2, j])+4*(raw_img[i, j-1]+raw_img[i, j+1])) / 8
                    res[i, j, 1] = raw_img[i, j]
                    res[i, j, 2] = (5*raw_img[i, j]-(raw_img[i-1, j-1]+raw_img[i-1, j+1]+raw_img[i+1, j-1]+raw_img[i+1, j+1] +
                                    raw_img[i-2, j]+raw_img[i+2, j])+0.5*(raw_img[i, j-2]+raw_img[i, j+2])+4*(raw_img[i-1, j]+raw_img[i+1, j])) / 8
                else:
                    res[i, j, 0] = raw_img[i, j]
                    res[i, j, 1] = (4*raw_img[i, j]-(raw_img[i-2, j]+raw_img[i+2, j]+raw_img[i, j-2]+raw_img[i, j+2])+2*(
                        raw_img[i-1, j]+raw_img[i+1, j]+raw_img[i, j-1]+raw_img[i, j+1])) / 8
                    res[i, j, 2] = (6*raw_img[i, j]+2*(raw_img[i-1, j-1]+raw_img[i-1, j+1]+raw_img[i+1, j-1] +
                                    raw_img[i+1, j+1])-1.5*(raw_img[i, j-2]+raw_img[i, j+2]+raw_img[i-2, j]+raw_img[i+2, j])) / 8
            else:
                if j % 2 == 0:
                    res[i, j, 0] = (6*raw_img[i, j]+2*(raw_img[i-1, j-1]+raw_img[i-1, j+1]+raw_img[i+1, j-1] +
                                    raw_img[i+1, j+1])-1.5*(raw_img[i, j-2]+raw_img[i, j+2]+raw_img[i-2, j]+raw_img[i+2, j])) / 8
                    res[i, j, 1] = (4*raw_img[i, j]-(raw_img[i-2, j]+raw_img[i+2, j]+raw_img[i, j-2]+raw_img[i, j+2])+2*(
                        raw_img[i-1, j]+raw_img[i+1, j]+raw_img[i, j-1]+raw_img[i, j+1])) / 8
                    res[i, j, 2] = raw_img[i, j]
                else:
                    res[i, j, 0] = (5*raw_img[i, j]-(raw_img[i-1, j-1]+raw_img[i-1, j+1]+raw_img[i+1, j-1]+raw_img[i+1, j+1] +
                                    raw_img[i-2, j]+raw_img[i+2, j])+0.5*(raw_img[i, j-2]+raw_img[i, j+2])+4*(raw_img[i-1, j]+raw_img[i+1, j])) / 8
                    res[i, j, 1] = raw_img[i, j]
                    res[i, j, 2] = (5*raw_img[i, j]-(raw_img[i-1, j-1]+raw_img[i-1, j+1]+raw_img[i+1, j-1]+raw_img[i+1, j+1] +
                                    raw_img[i, j-2]+raw_img[i, j+2])+0.5*(raw_img[i-2, j]+raw_img[i+2, j])+4*(raw_img[i, j-1]+raw_img[i, j+1])) / 8
    return np.array(res.clip(0, 255), dtype='uint8')


def compute_psnr(img_pred, img_gt):
    img_gt = np.array(img_gt, dtype='float64')
    img_pred = np.array(img_pred, dtype='float64')

    MSE = np.mean((img_gt - img_pred) ** 2)
    if MSE == 0:
        raise ValueError

    max = np.max(img_gt ** 2)
    return 10 * np.log10(max / MSE)
