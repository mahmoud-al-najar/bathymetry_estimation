import numpy as np
from scipy.signal import fftconvolve, detrend
from scipy.interpolate import interp1d
import copy


def __normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
        np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


def apply_normxcorr2(sub_tile):
    dim1_size = sub_tile.shape[0]
    dim2_size = sub_tile.shape[1]
    img = np.zeros(sub_tile.shape)
    for i in range(4):
        img_t = np.squeeze(sub_tile[:, :, i])
        img_t = img_t - np.median(img_t)
        c0 = __normxcorr2(img_t, img_t)
        c2 = __normxcorr2(img_t, c0)
        c2 = c2[int(c2.shape[0] / 2 - dim1_size / 2 + 1): int(c2.shape[0] / 2 + dim2_size / 2 + 1),
                int(c2.shape[0] / 2 - dim1_size / 2 + 1): int(c2.shape[0] / 2 + dim2_size / 2 + 1)]
        img[:, :, i] = c2
    return img


def apply_fft(sub_tile, t_max=25, t_min=5):
    n, m, c = sub_tile.shape
    kx = np.fft.fftshift(np.fft.fftfreq(n, 10))
    ky = np.fft.fftshift(np.fft.fftfreq(m, 10))
    kx = np.repeat(np.reshape(kx, (n, 1)), m, axis=1)
    ky = np.repeat(np.reshape(ky, (1, m)), n, axis=0)
    threshold_min = 1 / (1.56 * t_max ** 2)
    threshold_max = 1 / (1.56 * t_min ** 2)
    sub_tile = np.zeros(sub_tile.shape)
    for channel in range(c):
        r = sub_tile[:, :, channel]
        r = detrend(detrend(r, axis=1), axis=0)
        fftr = np.fft.fft2(r)
        energy_r = np.fft.fftshift(fftr)
        kr = np.sqrt(kx ** 2 + ky ** 2)
        kr[kr < threshold_min] = 0
        kr[kr > threshold_max] = 0
        bool_kr = (kr > 0)
        energy_r *= bool_kr
        sub_tile[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(energy_r)))
    return sub_tile


def apply_hanning(sub_tile):
    ri = np.arange(0, 1, 0.01)
    ai = (1 - (np.cos((np.pi * 0.5) + (np.pi * ri * 0.5)) ** 2))
    x_center = y_center = int(sub_tile.shape[1] / 2)
    tile_indices = np.indices(sub_tile.shape)
    dxmi = np.abs(tile_indices[2] - x_center)  # x is cross-shore
    dymi = np.abs(tile_indices[1] - y_center)
    r = np.sqrt((dxmi ** 2) + (dymi ** 2))
    r = r / np.max(r)
    wmi = interp1d(x=ri, y=ai, kind='linear', fill_value='extrapolate')(r)
    iw = wmi * sub_tile
    return iw


def apply_2d_gradient(sub_tile):
    """"tile: (cross_shore, long_shore, bands)"""
    cross_shore_slopes = copy.deepcopy(sub_tile)
    long_shore_slopes = copy.deepcopy(sub_tile)

    cross_shore_slopes = np.diff(cross_shore_slopes, axis=0)
    long_shore_slopes = np.diff(long_shore_slopes, axis=1)

    cross_shore_slopes = cross_shore_slopes ** 2
    long_shore_slopes = long_shore_slopes ** 2

    summed_up = long_shore_slopes[:-1, :, :] + cross_shore_slopes[:, :-1, :]
    return np.sqrt(summed_up)
