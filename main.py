import os

import imageio as iio
import numpy as np
import tifffile

from splicing_functions import splicing_functions
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Haar Wavelet blur detection with SVD on an image')
    parser.add_argument('-i', '--input_background', dest='input_background', type=str, required=True, help='image input background')
    parser.add_argument('-o', '--output_path', dest='output_path', type=str, default='output/', required=False, help="path to save output")
    parser.add_argument('-s', '--splicing_source', dest='splicing_source', type=str, help="splicing source")
    parser.add_argument('-m', '--splicing_mask', dest='splicing_mask', type=str, help="splicing mask")
    args = parser.parse_args()

    # Load image and add clouds to an image
    background = iio.imread(args.input_background).astype(np.float32)
    source = iio.imread(args.splicing_source).astype(np.float32)
    mask = tifffile.imread(args.splicing_mask).astype(np.float32)
    cloudy_image, mask = splicing_functions.splice_cloud(background, source, mask)

    # Save output
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    iio.imwrite(args.output_path + 'cloudy.tif', cloudy_image)
    iio.imwrite(args.output_path + 'background.tif', background)
    iio.imwrite(args.output_path + 'source.tif', source)
    iio.imwrite(args.output_path + 'mask.tif', mask)