import numpy as np


def compute_energy(image):
    image = np.array(image, dtype='float64')

    img_new = 0.299 * image[:, :, 0] + 0.587 * \
        image[:, :, 1] + 0.114 * image[:, :, 2]

    partial_x = np.zeros_like(img_new, dtype='float64')
    partial_y = np.zeros_like(img_new, dtype='float64')

    partial_x[1:-1, :] = (img_new[2:, :] - img_new[:-2, :]) / 2
    partial_y[:, 1:-1] = (img_new[:, 2:] - img_new[:, :-2]) / 2

    partial_x[0, :] = img_new[1, :] - img_new[0, :]
    partial_x[-1, :] = img_new[-1, :] - img_new[-2, :]

    partial_y[:, 0] = img_new[:, 1] - img_new[:, 0]
    partial_y[:, -1] = img_new[:, -1] - img_new[:, -2]

    return np.sqrt(partial_x ** 2 + partial_y ** 2)


def compute_seam_matrix(energy, mode, mask=None):
    res = np.zeros_like(energy, dtype='float64')
    big_num = energy.shape[1] * energy.shape[0] * 256
    width = energy.shape[1] - 1
    hight = energy.shape[0] - 1
    if mode == 'horizontal':
        res[0, :] = energy[0, :]
        if mask is not None:
            res[0, :] += big_num * mask[0, :]
        for i in range(1, energy.shape[0]):
            res[i, 0] = energy[i, 0] + min(res[i - 1, 0], res[i - 1, 1])
            if mask is not None:
                res[i, 0] += big_num * mask[i, 0]
            for j in range(1, width):
                res[i, j] = energy[i, j] + \
                    min(res[i-1, j-1], res[i-1, j], res[i-1, j+1])
                if mask is not None:
                    res[i, j] += big_num * mask[i, j]
            res[i, width] = energy[i, width] + \
                min(res[i - 1, width-1], res[i - 1, width])
            if mask is not None:
                res[i, width] += big_num * mask[i, width]
    elif mode == 'vertical':
        res[:, 0] = energy[:, 0]
        if mask is not None:
            res[:, 0] += big_num * mask[:, 0]
        for j in range(1, energy.shape[1]):
            res[0, j] = energy[0, j] + min(res[0, j-1], res[1, j-1])
            if mask is not None:
                res[0, j] += big_num * mask[0, j]
            for i in range(1, hight):
                res[i, j] = energy[i, j] + \
                    min(res[i-1, j-1], res[i, j-1], res[i+1, j-1])
                if mask is not None:
                    res[i, j] += big_num * mask[i, j]
            res[hight, j] = energy[hight, j] + \
                min(res[hight - 1, j-1], res[hight, j-1])
            if mask is not None:
                res[hight, j] += big_num * mask[hight, j]
    return res


def remove_minimal_seam(image, seam_matrix, mode, mask=None):
    width = image.shape[1]
    hight = image.shape[0]
    res_img = None
    res_mask = None
    shrink_mask = np.zeros_like(seam_matrix, dtype='uint8')
    if mode == 'horizontal shrink':
        res_img = np.zeros((hight, width-1, 3), dtype='uint8')
        if mask is not None:
            res_mask = np.zeros((hight, width-1), dtype='uint8')
        ind = np.argmin(seam_matrix[hight-1, :])
        for i in range(hight-1, 0, -1):
            res_img[i, :, :] = np.concatenate(
                (image[i, :ind, :], image[i, ind+1:, :]))
            if mask is not None:
                res_mask[i, :] = np.concatenate(
                    (mask[i, :ind], mask[i, ind+1:]))
            shrink_mask[i, ind] = 1
            if ind == 0:
                if seam_matrix[i-1, 0] > seam_matrix[i-1, 1]:
                    ind = 1
            elif ind == width-1:
                if seam_matrix[i-1, width-1] >= seam_matrix[i-1, width-2]:
                    ind = width - 2
            else:
                i_help = np.argmin(seam_matrix[i-1, ind-1:ind+2])
                if i_help == 0:
                    ind -= 1
                if i_help == 2:
                    ind += 1
        res_img[0, :, :] = np.concatenate(
            (image[0, :ind, :], image[0, ind+1:, :]))
        if mask is not None:
            res_mask[0, :] = np.concatenate(
                (mask[0, :ind], mask[0, ind+1:]))
        shrink_mask[0, ind] = 1
    else:
        res_img = np.zeros((hight-1, width, 3), dtype='uint8')
        if mask is not None:
            res_mask = np.zeros((hight-1, width), dtype='uint8')
        ind = np.argmin(seam_matrix[:, width-1])
        for i in range(width-1, 0, -1):
            res_img[:, i, :] = np.concatenate(
                (image[:ind, i, :], image[ind+1:, i, :]))
            if mask is not None:
                res_mask[:, i] = np.concatenate(
                    (mask[:ind, i], mask[ind+1:, i]))
            shrink_mask[ind, i] = 1
            if ind == 0:
                if seam_matrix[0, i-1] > seam_matrix[1, i-1]:
                    ind = 1
            elif ind == hight-1:
                if seam_matrix[hight-1, i-1] >= seam_matrix[hight-2, i-1]:
                    ind = hight - 2
            else:
                i_help = np.argmin(seam_matrix[ind-1:ind+2, i-1])
                if i_help == 0:
                    ind -= 1
                if i_help == 2:
                    ind += 1
        res_img[:, 0, :] = np.concatenate(
            (image[:ind, 0, :], image[ind+1:, 0, :]))
        if mask is not None:
            res_mask[:, 0] = np.concatenate(
                (mask[:ind, 0], mask[ind+1:, 0]))
        shrink_mask[ind, 0] = 1
    return res_img, res_mask, shrink_mask


def seam_carve(image, mode, mask=None):
    energy = compute_energy(image)

    mode_to_func = 'horizontal' if mode == 'horizontal shrink' else 'vertical'
    seam_matrix = compute_seam_matrix(energy, mode_to_func, mask)

    return remove_minimal_seam(image, seam_matrix, mode, mask)
