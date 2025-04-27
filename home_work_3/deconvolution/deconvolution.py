import numpy as np
from scipy import fft


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """

    center = (size - 1.0) / 2

    x = (np.arange(size) - center)[:, None]
    y = np.arange(size) - center
    res = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / \
        (2 * np.pi * sigma ** 2)

    return res / np.sum(res)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    th, tw = shape[0], shape[1]
    kh, kw = h.shape[:2]
    ph, pw = th - kh, tw - kw

    padding = [((ph+1) // 2, ph // 2), ((pw+1) // 2, pw // 2)]
    kernel = np.pad(h, padding)

    kernel = np.fft.ifftshift(kernel)
    return fft.fft2(kernel)


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    res = np.zeros_like(H)
    res[np.abs(H) > threshold] = 1 / H[np.abs(H) > threshold]
    return res


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    g = fft.fft2(blurred_img)
    h_inv = inverse_kernel(fourier_transform(h, blurred_img.shape), threshold)
    res = fft.ifft2(g * h_inv)

    return np.abs(res)


def wiener_filtering(blurred_img, h, K=0.00005):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    ker = fourier_transform(h, blurred_img.shape)

    g = fft.fft2(blurred_img)
    res = fft.ifft2(g * np.conjugate(ker) / (np.conjugate(ker) * ker + K))

    return np.abs(res)


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    return 20 * np.log10(255 / np.sqrt(np.mean((img1 - img2) ** 2)))
