import numpy as np
import tifffile
import imageio as iio
import random

from numpy import ma

import matplotlib.pyplot as plt

# ============= ALPHA MATTING ================== #
"""Guided filter for masked arrays"""
import functools

import numba as nb
import numpy.ma as ma


# @nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32, nb.float32[:]))
def conv(u, radius, kernel):
    """This is much faster than scipy.ndimage.convolve"""
    kernel = np.ones((2 * radius + 1,), dtype=np.float32) / (2 * radius + 1)
    for n in (0, 1):
        # ravel ensures that the array is contiguous (array is copied if
        # needed). Default order for ravel and reshape is C.
        u = np.pad(u, ((0, 0), (radius, radius)), 'symmetric')
        u = np.convolve(u.ravel(), kernel, 'same').reshape(u.shape)
        u = u[:, radius:-radius]
        u = u.transpose((1, 0))
    return np.ascontiguousarray(u)


def conv_masked(u, *args):
    assert isinstance(u, ma.MaskedArray), type(u)
    u_data = u.filled(0).astype(np.float32)
    u_mask = (~u.mask).astype(np.float32)
    return ma.array(conv(u_data, *args) / conv(u_mask, *args), mask=u.mask)


def filter_with_guide(image, guide, radius, epsilon):
    """
    """
    # radius = 128
    conv_masked_partial = lambda u: conv_masked(u, radius, epsilon)

    image = image.astype(np.float32)
    guide = guide.astype(np.float32)

    g_mean = conv_masked_partial(guide)
    i_mean = conv_masked_partial(image)
    g_var = conv_masked_partial(guide * guide) - g_mean * g_mean
    gi_cov = conv_masked_partial(guide * image) - g_mean * i_mean

    a = gi_cov / (g_var + epsilon ** 2)
    b = i_mean - a * g_mean

    a_mean = conv_masked_partial(a)
    b_mean = conv_masked_partial(b)

    return a_mean * guide + b_mean


def translation_mask(mask, dx, dy, shape_out):
    h, w = mask.shape[:2]
    dx = int(dx)
    dy = int(dy)
    ts_mat = np.array([[1, 0, dx], [0, 1, dy]])

    translated_mask = np.zeros(shape_out)
    for i in range(h):
        for j in range(w):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x, origin_y, 1])

            new_xy = np.dot(ts_mat, origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < w and 0 < new_y < h:
                translated_mask[new_y, new_x] = mask[i, j]

    return translated_mask


def add_shadows(img, mask, radius, theta, light_intensity):
    delta_x = radius * np.cos(theta)
    delta_y = radius * np.sin(theta)

    translated_mask = translation_mask(mask, delta_x, delta_y, mask.shape)

    # Add shadows
    dark = img * light_intensity
    shadows = (translated_mask * 2) * dark + (1 - translated_mask) * img

    return shadows


def splice_cloud(background, cloud_image, mask):

    if len(background.shape) > 2:
        cloudy_background = []
        for channel_idx in range(background.shape[2]):
            bg_channel = background[:, :, channel_idx]
            cloud_channel = cloud_image[:, :, channel_idx]

            mask_cloud_conv = (conv(mask, 5, None) > 0.9).astype(np.float32)
            cloud_channel_ma, mask_cloud_conv = ma.array(cloud_channel, mask=np.zeros_like(mask_cloud_conv)), ma.array(
                mask_cloud_conv,
                mask=np.zeros_like(mask_cloud_conv))
            cloud_alpha = filter_with_guide(mask_cloud_conv, cloud_channel_ma, 16, 0.01)
            cloud = (cloud_alpha.data - np.min(cloud_alpha.data)) / (
                    np.max(cloud_alpha.data) - np.min(cloud_alpha.data)) * 1.2

            # Add some shadows before adding clouds
            shadows_bg = add_shadows(bg_channel, cloud, 100, np.pi / 4, 0.05)

            # Copy the cloud into the background image
            cloudy_channel = cloud_channel * ((cloud-0.3) * 2) + shadows_bg * (1 - cloud)
            cloudy_background.append(cloudy_channel)
        cloudy_background = np.stack(cloudy_background, axis=2)

    else:
        bg_channel = background
        cloud_channel = cloud_image

        mask_cloud_conv = (conv(mask, 5, None) > 0.9).astype(np.float32)
        cloud_channel_ma, mask_cloud_conv = ma.array(cloud_channel, mask=np.zeros_like(mask_cloud_conv)), ma.array(
            mask_cloud_conv,
            mask=np.zeros_like(mask_cloud_conv))
        cloud_alpha = filter_with_guide(mask_cloud_conv, cloud_channel_ma, 16, 0.01)
        cloud = (cloud_alpha.data - np.min(cloud_alpha.data)) / (
                np.max(cloud_alpha.data) - np.min(cloud_alpha.data)) * 1.2

        # Add some shadows before adding clouds
        shadows_bg = add_shadows(bg_channel, cloud, 100, np.pi / 4, 0.05)

        # Copy the cloud into the background image
        cloudy_background = cloud_channel * cloud + shadows_bg * (1 - cloud)

    return cloudy_background, mask

