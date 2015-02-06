#!/usr/bin/env python

"""
   Written by Joerg Dietrich <astro@joergdietrich.com>. Copyright 2015
   Based on original code by Oliver Siemoneit. Copyright 2007
   This code is licensed under the GNU GPL version 2, see COPYING for details.
"""

from __future__ import print_function, division

from collections import OrderedDict
import os
try:
    import pickle
except ImportError:
    import cPickle as pickle
    
from PIL import Image
import numpy as np
try:
    import matplotlib as mpl
    _no_mpl = False
except ImportError:
    _no_mpl = True
    

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
    # rgb - sim_rgb contains the color information that dichromats
    # cannot see. err2mod rotates this to a part of the spectrum that
    # they can see.
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


def get_child_colors(child, mpl_colors):
    """
    Recursively enter all colors of a matplotlib objects and its
    children into a dictionary.
    
    Arguments:
    ----------
    child : a matplotlib object
    mpl_colors : OrderedDict from collections

    Returns:
    --------
    mpl_colors : OrderedDict
    """
    mpl_colors[child] = OrderedDict()
    # does not deal with cmaps yet, only with lines, patches,
    # etc. (vector like stuff)
    if hasattr(child, "get_color"):
        mpl_colors[child]['color'] = child.get_color()
    if hasattr(child, "get_facecolor"):
        mpl_colors[child]['fc'] = child.get_facecolor()
    if hasattr(child, "get_edgecolor"):
        mpl_colors[child]['ec'] = child.get_edgecolor()
    if hasattr(child, "get_markeredgecolor"):
        mpl_colors[child]['mec'] = child.get_markeredgecolor()
    if hasattr(child, "get_markerfacecolor"):
        mpl_colors[child]['mfc'] = child.get_markerfacecolor()
    if hasattr(child, "get_markerfacecoloralt"):
        mpl_colors[child]['mfcalt'] = child.get_markerfacecoloralt()
    if hasattr(child, "get_children"):
        grandchildren = child.get_children()
        for grandchild in grandchildren:
            mpl_colors = get_child_colors(grandchild, mpl_colors)
    return mpl_colors


def get_mpl_colors(fig):
    """
    Read all colors used in a matplotlib figure into an OrderedDict.

    Arguments:
    ----------
    fig : matplotlib.figure.Figure

    Returns:
    --------
    mpl_dict : OrderedDict from collections
    """
    mpl_colors = OrderedDict()
    children = fig.get_children()
    for child in children:
        mpl_colors = get_child_colors(child, mpl_colors)
    return mpl_colors


def get_key_colors(mpl_colors, rgb, alpha):
    if _no_mpl == True:
        raise ImportError("matplotlib not found, " \
                          "can only deal with pixel images")
    cc = mpl.colors.ColorConverter()
    # Note that the order must match the insertion order in
    # get_child_colors()
    color_keys = ("color", "fc", "ec", "mec", "mfc", "mfcalt")
    for color_key in color_keys:
        try:
            color = mpl_colors[color_key]
            # skip unset colors, otherwise they are turned into black.
            if color == 'none':
                continue
            rgba = cc.to_rgba_array(color)
            rgb = np.append(rgb, rgba[:, :3])
            alpha = np.append(alpha, rgba[:, 3])
        except KeyError:
            pass
        for key in mpl_colors.keys():
            if key in color_keys:
                continue
            rgb, alpha = get_key_colors(mpl_colors[key], rgb, alpha)
    return rgb, alpha


def arrays_from_dict(mpl_colors):
    """
    Create rgb and alpha arrays from color dictionary.

    Arguments:
    ----------
    mpl_colors : OrderedDict
        dictionary with all colors of all children, matplotlib instances are
        keys

    Returns:
    --------
    rgb : array of shape (M, 1, 3)
        RGB values of colors in a line image, M is the total number of 
        non-unique colors
    alpha : array of shape (M, 1)
        Alpha channel values of all mpl instances
    """
    rgb = np.array([])
    alpha = np.array([])
    for key in mpl_colors.keys():
        rgb, alpha = get_key_colors(mpl_colors[key], rgb, alpha)
    m = rgb.size / 3
    rgb = rgb.reshape((m, 1, 3))
    return rgb, alpha


def set_mpl_colors(mpl_colors, rgba, i=0):
    """
    """
    cc = mpl.colors.ColorConverter()
    # Note that the order must match the insertion order in
    # get_child_colors()
    color_keys = ("color", "fc", "ec", "mec", "mfc", "mfcalt")
    for key in mpl_colors.keys():
        if key in color_keys:
            continue
        for color_key in color_keys:
            try:
                color = mpl_colors[key][color_key]
                # skip unset colors, otherwise they are turned into black.
                if color == 'none':
                    continue
                color_shape = cc.to_rgba_array(color).shape
                j = color_shape[0]
                target_color = rgba[i:i+j, :].reshape(j, 4)
                target_color = target_color[0]
                i += j
                if color_key == "color":
                    key.set_color(target_color)
                elif color_key == "fc":
                    key.set_facecolor(target_color)
                elif color_key == "ec":
                    key.set_edgecolor(target_color)
                elif color_key == "mec":
                    key.set_markeredgecolor(target_color)
                elif color_key == "mfc":
                    key.set_markerfacecolor(target_color)
                elif color_key == "mfcalt":
                    key.set_markerfacecoloralt(target_color)
            except KeyError:
                pass
        for key in mpl_colors.keys():
            i = set_mpl_colors(mpl_colors[key], rgba, i)
    return i

        
def simulate_mpl(fig, color_deficit='d', copy=False):
    """
    fig : matplotlib.figure.Figure
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    copy : bool, optional
        should simulation happen on a copy (True) or the original 
        (False, default)

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    if copy:
        # mpl.transforms cannot be copy.deepcopy()ed. Thus we resort
        # to pickling.
        # Turns out PolarAffine cannot be unpickled ...
        pfig = pickle.dumps(fig)
        fig = pickle.loads(pfig)
    mpl_colors = get_mpl_colors(fig)
    rgb, alpha = arrays_from_dict(mpl_colors)
    sim_rgb = simulate(array_to_img(rgb), color_deficit)
    # clip values to lie in the range [0, 1]
    comp_arr1 = np.zeros_like(sim_rgb)
    comp_arr2 = np.ones_like(sim_rgb)
    sim_rgb = np.maximum(comp_arr1, sim_rgb)
    sim_rgb = np.minimum(comp_arr2, sim_rgb)

    r, g, b = np.split(sim_rgb, 3, 2)
    rgba = np.concatenate((r, g, b, alpha.reshape(alpha.size, 1, 1)),
                           axis=2).reshape(-1, 4)
    set_mpl_colors(mpl_colors, rgba)
    fig.canvas.draw()
    return fig
    
    
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
    if args.type is None:
        args.type = "d"

    orig_img = Image.open(args.input_image)
    sim_rgb, rgb = simulate(orig_img, args.type, return_original_rgb=True)
    if args.simulate:
        sim_img = array_to_img(sim_rgb)
        sim_img.save(args.output_image)
    if args.daltonize:
        dalton_rgb = daltonize(rgb, sim_rgb)
        dalton_img = array_to_img(dalton_rgb)
        dalton_img.save(args.output_image)
