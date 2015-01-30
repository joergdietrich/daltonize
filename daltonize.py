#!/usr/bin/env python

"""
   Written by Joerg Dietrich <astro@joergdietrich.com>. Copyright 2015
   Based on original code by Oliver Siemoneit. Copyright 2007
   This code is licensed under the GNU GPL version 2, see COPYING for details.
"""

from __future__ import print_function, division

import os

from PIL import Image
import numpy as np


def transform_colorspace(img, mat):
    """Transform image to a different color space.

    Arguments:
    ----------
    img : array of shape (M, N, 3)
    mat : array of shape (3, 3)
        conversion matrix to different color space

    Returns:
    --------
    out : array of shape (M, N, 3)
    """
    # Fast element (=pixel) wise matrix multiplication
    return np.einsum("ij, ...j", mat, img)


def simulate(img, color_deficit="d", return_original_rgb=False):
    """Simulate the effet of colorblindness on an image.

    Arguments:
    ----------
    img : PIL.PngImagePlugin.PngImageFile, input image
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    return_original_rgb : bool, optional
        Return the original image as rgb if True, default False

    Returns:
    --------
    sim_rgb : array of shape (M, N, 3)
        simulated image in RGB format
    rgb : array of shape (M, N, 3)
        original image in RGB format. Returned only if return_original_rgb is
        True
    """
    # Colorspace transformation matrices
    cb_matrices = {
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
        "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
    }
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    # Precomputed inverse
    lms2rgb = np.array([[8.09444479e-02, -1.30504409e-01,  1.16721066e-01],
                        [-1.02485335e-02,  5.40193266e-02, -1.13614708e-01],
                        [-3.65296938e-04, -4.12161469e-03,  6.93511405e-01]])

    img = img.copy()
    img = img.convert('RGB')

    rgb = np.asarray(img, dtype=float)
    # first go from RBG to LMS space
    lms = transform_colorspace(rgb, rgb2lms)
    # Calculate image as seen by the color blind
    sim_lms = transform_colorspace(lms, cb_matrices[color_deficit])
    # Transform back to RBG
    sim_rgb = transform_colorspace(sim_lms, lms2rgb)
    if return_original_rgb:
        return sim_rgb, rgb
    return sim_rgb


def daltonize(rgb, sim_rgb):
    """
    Adjust color palette of an image to compensate color blindness.

    Arguments:
    ----------
    rgb : array of shape (M, N, 3)
        original image in RGB format
    sim_rgb : array of shape (M, N, 3)
        image with simulated color blindness

    Returns:
    dtpn : array of shape (M, N, 3)
        image in RGB format with colors adjusted
    """
    err2mod = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    err = transform_colorspace(rgb - sim_rgb, err2mod)
    dtpn = err + rgb
    return dtpn


def array_to_img(arr):
    """Convert a numpy array to a PIL image.

    Arguments:
    ----------
    arr : array of shape (M, N, 3)

    Returns:
    --------
    img : PIL.Image.Image
        RGB image created from array
    """
    # clip values to lie in the range [0, 255]
    comp_arr1 = np.zeros_like(arr)
    comp_arr2 = np.ones_like(arr) * 255
    arr = np.maximum(comp_arr1, arr)
    arr = np.minimum(comp_arr2, arr)
    arr = arr.astype('uint8')
    img = Image.fromarray(arr, mode='RGB')
    return img


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("output_image", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--simulate", help="create simulated image",
                        action="store_true")
    group.add_argument("-d", "--daltonize",
                        help="adjust image color palette for color blindness",
                       action="store_true")
    parser.add_argument("-t", "--type", type=str, choices=["d", "p", "t"],
                        help="type of color blindness (deuteranopia, "
                        "protanopia, tritanopia), default is deuteranopia "
                        "(most common)")
    args = parser.parse_args()

    if args.simulate is False and args.daltonize is False:
        print("No action specified, assume daltonizing")
        args.daltonize = True
        
    orig_img = Image.open(args.input_image)
    sim_rgb, rgb = simulate(orig_img, args.type, return_original_rgb=True)
    if args.simulate:
        sim_img = array_to_img(sim_rgb)
        sim_img.save(args.output_image)
    if args.daltonize:
        dalton_rgb = daltonize(rgb, sim_rgb)
        dalton_img = array_to_img(dalton_rgb)
        dalton_img.save(args.output_image)
