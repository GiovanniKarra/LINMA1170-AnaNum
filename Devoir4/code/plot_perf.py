import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import path, remove

from image_svd import compress_image, image_to_matrix


def plot_compress_perf(filepath, compression_min, compression_max, N, title=""):
	levels = np.logspace(np.log10(compression_min), np.log10(compression_max), N)
	sizes = np.empty(N)

	inital_size = path.getsize(filepath)

	matrix = image_to_matrix(filepath)

	for i, level in enumerate(levels):
		compress_image(matrix, "temp.jpg", level)
		sizes[i] = path.getsize("temp.jpg")
		print("done %d, size : %f"%(i, sizes[i]))

	remove("temp.jpg")
	
	plt.figure()

	plt.title("Compression efficency (%s)"%title)

	plt.xlabel("compression level")
	plt.ylabel("size ratio")

	plt.semilogx(levels, sizes/inital_size)

	plt.grid(which="major", linestyle="-")
	plt.grid(which="minor", linestyle=":")

	plt.legend([])

	# plt.show()
	plt.savefig("images/%s_plot.svg"%title, format="svg")


if __name__ == "__main__":
	# plot_compress_perf("images/Jean-Francois_Remacle.jpg", 1, 1000, 30, title="Jean-François Remacle")

	A = np.asarray(np.random.rand(1000, 1000, 3)*255, dtype=np.uint8)
	print(A)
	image = Image.fromarray(A, "RGB")
	image.save("images/random.jpg")

	plot_compress_perf("images/random.jpg", 1, 1000, 30, title="Random Image")
