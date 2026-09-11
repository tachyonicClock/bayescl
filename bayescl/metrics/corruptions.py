"""ImageNet-C-style synthetic distribution shift (EXPERIMENT.md §4.2).

Source: https://github.com/bethgelab/imagecorruptions (Apache-2.0), by
Michaelis et al., reimplementing Hendrycks & Dietterich's ImageNet-C
corruptions. Trimmed to the subset of the 15 "common" corruption types that
need neither ``opencv-python`` (whose GUI build requires a system ``libGL``
we have no way to install) nor ``numba`` (a heavy JIT dependency needed by
only one corruption, ``glass_blur``). The remaining functions are otherwise
unmodified aside from replacing the deprecated
``scipy.ndimage.interpolation`` import path.

Dropped relative to the full 15: ``defocus_blur``, ``frost``, ``snow``
(all need ``cv2``) and ``glass_blur`` (needs ``numba``).
"""

import math
from io import BytesIO

import numpy as np
import skimage as sk
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian


def clipped_zoom(img, zoom_factor):
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(
        img[top0 : top0 + ch0, top1 : top1 + ch1],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    return img


def _get_optimal_kernel_width_1d(radius, sigma):
    return radius * 2 + 1


def _gauss_function(x, mean, sigma):
    return (np.exp(-((x - mean) ** 2) / (2 * (sigma**2)))) / (
        np.sqrt(2 * np.pi) * sigma
    )


def _get_motion_blur_kernel(width, sigma):
    k = _gauss_function(np.arange(width), 0, sigma)
    return k / np.sum(k)


def _shift(image, dx, dy):
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1 : dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx : dx + 1]
    else:
        shifted = image

    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1 : dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy : dy + 1, :]
    return shifted


def _motion_blur(x, radius, sigma, angle):
    width = _get_optimal_kernel_width_1d(radius, sigma)
    kernel = _get_motion_blur_kernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            break
        blurred = blurred + kernel[i] * _shift(x, dx, dy)
    return blurred


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _plasma_fractal(mapsize=256, wibbledecay=3):
    """Diamond-square heightmap generator, side length a power of two."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float32)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize_ : stepsize, stepsize // 2 : mapsize_ : stepsize
        ]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize, stepsize // 2 : mapsize_ : stepsize] = (
            wibbledmean(ltsum)
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize_ : stepsize, 0:mapsize_:stepsize] = (
            wibbledmean(ttsum)
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def gaussian_noise(x, severity=1):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def motion_blur(x, severity=1):
    shape = np.array(x).shape
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    x = np.array(x)

    angle = np.random.uniform(-45, 45)
    x = _motion_blur(x, radius=c[0], sigma=c[1], angle=angle)

    if len(x.shape) < 3 or x.shape[2] < 3:
        gray = np.clip(np.array(x).transpose((0, 1)), 0, 255)
        if len(shape) >= 3 or shape[2] >= 3:
            return np.stack([gray, gray, gray], axis=2)
        return gray
    return np.clip(x, 0, 255)


def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)

    set_exception = False
    for zoom_factor in c:
        if len(x.shape) < 3 or x.shape[2] < 3:
            x_channels = np.array([x, x, x]).transpose((1, 2, 0))
            zoom_layer = clipped_zoom(x_channels, zoom_factor)
            zoom_layer = zoom_layer[: x.shape[0], : x.shape[1], 0]
        else:
            zoom_layer = clipped_zoom(x, zoom_factor)
            zoom_layer = zoom_layer[: x.shape[0], : x.shape[1], :]

        try:
            out += zoom_layer
        except ValueError:
            set_exception = True
            out[: zoom_layer.shape[0], : zoom_layer.shape[1]] += zoom_layer

    if set_exception:
        print("ValueError for zoom blur, exception handling")
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]

    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = _next_power_of_2(int(max_side))

    x = np.array(x) / 255.0
    max_val = x.max()

    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * _plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
            : shape[0], : shape[1]
        ]
    else:
        x += (
            c[0]
            * _plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                : shape[0], : shape[1]
            ][..., np.newaxis]
        )
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = np.array(x) / 255.0

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    gray_scale = False
    if x.mode != "RGB":
        gray_scale = True
        x = x.convert("RGB")
    x.save(output, "JPEG", quality=c)
    x = Image.open(output)
    if gray_scale:
        x = x.convert("L")
    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x_shape = np.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)
    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)
    return x


def elastic_transform(image, severity=1):
    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    sigma = np.array(shape_size) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][severity - 1]
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005

    dx = (
        gaussian(
            np.random.uniform(-max_dx, max_dx, size=shape[:2]),
            sigma,
            mode="reflect",
            truncate=3,
        )
        * alpha
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-max_dy, max_dy, size=shape[:2]),
            sigma,
            mode="reflect",
            truncate=3,
        )
        * alpha
    ).astype(np.float32)

    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(
            np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2])
        )
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        * 255
    )


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}


def corruption_names() -> list[str]:
    return list(CORRUPTIONS)


def corrupt(image: Image.Image, corruption_name: str, severity: int) -> Image.Image:
    """Apply one ImageNet-C-style corruption to a PIL image at a severity in [1, 5]."""
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in [1, 5], got {severity}")
    out = CORRUPTIONS[corruption_name](image, severity)
    if isinstance(out, Image.Image):
        return out
    return Image.fromarray(np.uint8(out))
