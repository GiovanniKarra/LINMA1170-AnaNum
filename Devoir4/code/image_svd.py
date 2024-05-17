#Pierre et Giovanni Karrra 
import numpy as np
from numpy.linalg import svd
from PIL import Image


def compress_image(image_matrix, output_path, compression_level):

	m, n, _ = image_matrix.shape
	k = int(min(m, n)/compression_level)

	R = image_matrix[:, :, 0]
	G = image_matrix[:, :, 1]
	B = image_matrix[:, :, 2]

	U, S, VT = svd(R, full_matrices=False)
	R = (U[:, :k] * S[:k]) @ VT[:k, :]

	U, S, VT = svd(G, full_matrices=False)
	G = (U[:, :k] * S[:k]) @ VT[:k, :]

	U, S, VT = svd(B, full_matrices=False)
	B = (U[:, :k] * S[:k]) @ VT[:k, :]

	compressed_matrix = np.stack([R, G, B], axis=2, dtype=np.int8, casting="unsafe")

	image = Image.fromarray(compressed_matrix, "RGB")
	image.save(output_path)
	# image.show()


def image_to_matrix(filepath):
	image = Image.open(filepath)

	return np.asarray(image)