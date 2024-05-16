import numpy as np
from numpy.linalg import svd
from PIL import Image
import argparse

from image_svd import compress_image


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="SVD Compressor",
		usage="py compress.py [filename] [compression level] [OPTIONAL: -o output_file]",
		description="Compress an image using SVD decomposition"
	)
	parser.add_argument("filename")
	parser.add_argument("compression_level")
	parser.add_argument("-o", "--output", required=False, default=None)

	args = parser.parse_args()

	filename = args.filename
	compression_level = int(args.compression_level)
	output = args.output\
		if args.output is not None\
			else filename+"compressed.jpg"

	image = Image.open(filename)

	image_matrix = np.asarray(image)

	compress_image(image_matrix, output, compression_level)