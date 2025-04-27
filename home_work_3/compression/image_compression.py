import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here

    # Отцентруем каждую строчку матрицы
    means = np.mean(matrix, axis=1)
    matrix = matrix - means[:, None]

    # Найдем матрицу ковариации
    cov = np.cov(matrix)

    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigen_val, eigen_vec = np.linalg.eigh(cov)

    # Сортируем собственные значения в порядке убывания
    idx = np.flip(np.argsort(eigen_val)[-p:])

    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eigen_vec = eigen_vec[:, idx]

    # Проекция данных на новое пространство
    proection = eigen_vec.T.dot(matrix)

    return eigen_vec, proection, means


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        eigen_vec, proection, means = comp
        matrix = eigen_vec.dot(proection) + means[:, None]
        result_img.append(
            np.array(np.clip(np.round(matrix), 0, 255), dtype='uint8'))
        # Your code here

    return np.dstack(result_img)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            compressed.append(pca_compression(img[..., j], p))

        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.1687, -0.3313, 0.5],
                       [0.5, -0.4187, -0.0813]])

    vector = np.array([0, 128, 128])

    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i, j] = vector + matrix.dot(img[i, j])

    return np.array(np.clip(np.round(res), 0, 255), dtype='uint8')


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    matrix = np.array([[1, 0, 1.402],
                       [1, -0.34414, -0.71414],
                       [1, 1.77, 0]])

    vector = np.array([0, 128, 128])

    res = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i, j] = matrix.dot(img[i, j] - vector)

    return np.array(np.clip(np.round(res), 0, 255), dtype='uint8')


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    ycbcr_img = rgb2ycbcr(rgb_img)

    ycbcr_img[:, :, 1] = np.clip(
        np.round(gaussian_filter(ycbcr_img[:, :, 1], 10)), 0, 255)
    ycbcr_img[:, :, 2] = np.clip(
        np.round(gaussian_filter(ycbcr_img[:, :, 2], 10)), 0, 255)
    plt.imshow(ycbcr2rgb(ycbcr_img))

    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)

    ycbcr_img[:, :, 0] = np.clip(
        np.round(gaussian_filter(ycbcr_img[:, :, 0], 10)), 0, 255)

    plt.imshow(ycbcr2rgb(ycbcr_img))

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    component = gaussian_filter(component, 10)
    hight = component.shape[0] // 2 + 1
    width = component.shape[1] // 2 + 1
    mask = np.tile([[True, False],
                    [False, False]], (hight, width))[:-(2-component.shape[0] % 2), :-(2-component.shape[1] % 2)]

    return component[mask].reshape((component.shape[0] // 2 + component.shape[0] % 2, component.shape[1] // 2 + component.shape[1] % 2))


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    res = np.zeros((8, 8), dtype='float64')
    for u in range(8):
        for v in range(8):
            x = (2*np.arange(8) + 1) * u * np.pi / 16
            y = (2*np.arange(8) + 1) * v * np.pi / 16
            res[u, v] = (block * (np.cos(x)[:, None] * np.cos(y))).sum() / 4
            if u == 0:
                res[u, v] /= np.sqrt(2)
            if v == 0:
                res[u, v] /= np.sqrt(2)

    return res


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    if 1 <= q < 50:
        s = 5000 / q
    if 50 <= q < 100:
        s = 200 - 2 * q
    if q == 100:
        s = 1
    res = np.floor((50 + s * default_quantization_matrix) / 100)
    res[res == 0] = 1

    return res


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    res = np.zeros(64)
    count = 0
    i, j = 0, 0
    up = True

    while count < 64:
        res[count] = block[i, j]
        if up:
            if i == 0:
                j += 1
                up = False
            elif j == 7:
                i += 1
                up = False
            else:
                i -= 1
                j += 1
        else:
            if i == 7:
                j += 1
                up = True
            elif j == 0:
                i += 1
                up = True
            else:
                i += 1
                j -= 1
        count += 1

    return res


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    res = []
    count = 0
    for i in zigzag_list:
        if i != 0:
            if count != 0:
                res.append(0)
                res.append(count)
                count = 0
            res.append(i)
        else:
            count += 1
    if count != 0:
        res.append(0)
        res.append(count)
    return res


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Переходим из RGB в YCbCr
    ycbcr_img = rgb2ycbcr(img)

    small_img_color = np.zeros((img.shape[0] // 2, img.shape[1] // 2, 2))

    # Уменьшаем цветовые компоненты
    for i in range(2):
        # print(small_img_color[..., i].shape)
        small_img_color[..., i] = np.clip(
            np.round(downsampling(ycbcr_img[..., i + 1])), 0, 255)
    small_img_color = np.array(small_img_color, dtype='uint8')
    res_y = []
    res_cb = []
    res_cr = []
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    for i in range(small_img_color.shape[0] // 8):
        for j in range(small_img_color.shape[1] // 8):
            block = small_img_color[8*i:8*(i+1), 8*j:8*(j+1), :] - 128
            # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
            res_cb.append(compression(
                zigzag(quantization(dct(block[..., 0]), quantization_matrixes[1]))))
            res_cr.append(compression(
                zigzag(quantization(dct(block[..., 1]), quantization_matrixes[1]))))
    for i in range(ycbcr_img.shape[0] // 8):
        for j in range(ycbcr_img.shape[1] // 8):
            block = ycbcr_img[8*i:8*(i+1), 8*j:8*(j+1), 0]  # - 128
            # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
            res_y.append(compression(
                zigzag(quantization(dct(block), quantization_matrixes[0]))))

    return [res_y, res_cb, res_cr]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    res = []
    j = 0
    while j < len(compressed_list):
        if compressed_list[j] != 0:
            res.append(compressed_list[j])
            j += 1
        else:
            for i in range(compressed_list[j + 1]):
                res.append(0)
            j += 2
    return res


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    res = np.zeros((8, 8))
    count = 0
    i, j = 0, 0
    up = True

    while count < 64:
        res[i, j] = input[count]
        if up:
            if i == 0:
                j += 1
                up = False
            elif j == 7:
                i += 1
                up = False
            else:
                i -= 1
                j += 1
        else:
            if i == 7:
                j += 1
                up = True
            elif j == 0:
                i += 1
                up = True
            else:
                i += 1
                j -= 1
        count += 1

    return res


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    res = np.zeros((8, 8), dtype='float64')
    helper = np.ones((8, 8))
    helper[0, :] /= np.sqrt(2)
    helper[:, 0] /= np.sqrt(2)
    for x in range(8):
        for y in range(8):
            u = (2*x + 1) * np.arange(8) * np.pi / 16
            v = (2*y + 1) * np.arange(8) * np.pi / 16
            res[x, y] = (helper * block * (np.cos(u)
                         [:, None] * np.cos(v))).sum() / 4

    return np.round(res)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    res = np.zeros(
        (component.shape[0] * 2, component.shape[1] * 2), dtype='uint8')
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = component[i // 2, j // 2]

    return res


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    blocks_y = []
    blocks_cb = []
    blocks_cr = []

    y, cb, cr = result[0], result[1], result[2]

    for compr_list in y:
        blocks_y.append(inverse_dct(inverse_quantization(inverse_zigzag(
            inverse_compression(compr_list)), quantization_matrixes[0])))

    for compr_list in cb:
        blocks_cb.append(inverse_dct(inverse_quantization(inverse_zigzag(
            inverse_compression(compr_list)), quantization_matrixes[1])))

    for compr_list in cr:
        blocks_cr.append(inverse_dct(inverse_quantization(inverse_zigzag(
            inverse_compression(compr_list)), quantization_matrixes[1])))

    y_component = np.zeros((result_shape[0], result_shape[1]))
    cb_component = np.zeros((result_shape[0] // 2, result_shape[1] // 2))
    cr_component = np.zeros((result_shape[0] // 2, result_shape[1] // 2))

    count = 0
    for i in range(result_shape[0] // 8):
        for j in range(result_shape[1] // 8):
            y_component[8 * i:8*(i+1), 8*j:8*(j+1)] = blocks_y[count]  # + 128
            count += 1

    count = 0
    for i in range(cb_component.shape[0] // 8):
        for j in range(cb_component.shape[1] // 8):
            cb_component[8 * i:8*(i+1), 8*j:8*(j+1)] = blocks_cb[count] + 128
            count += 1

    count = 0
    for i in range(cr_component.shape[0] // 8):
        for j in range(cr_component.shape[1] // 8):
            cr_component[8 * i:8*(i+1), 8*j:8*(j+1)] = blocks_cr[count] + 128
            count += 1

    cb_component = upsampling(cb_component)
    cr_component = upsampling(cr_component)

    res = ycbcr2rgb(np.dstack([y_component, cb_component, cr_component]))

    return res


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        y_quant = own_quantization_matrix(y_quantization_matrix, p)
        color_quant = own_quantization_matrix(color_quantization_matrix, p)
        res_shape = img.shape
        img_compressed = jpeg_compression(img, [y_quant, color_quant])

        axes[i // 3, i % 3].imshow(jpeg_decompression(img_compressed,
                                   res_shape, [y_quant, color_quant]))
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(
            color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append(
                (pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(
            img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    np.savez_compressed(os.path.join('tmp', 'tmp.npz'),
                        np.array(compressed, dtype=np.object_))
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [
                       1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")
