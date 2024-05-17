#take an image remacle.png as input and show its compression with different factors

import numpy as np
from numpy.linalg import svd
from PIL import Image
import argparse
import os

from image_svd import compress_image
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
#Seaborn theme
sns.set_theme(style="whitegrid")
mpl.use('TkAgg')

def compress_and_display(filename, compression_level):
    image = Image.open(filename)
    plt.figure()
    
    image_matrix = np.asarray(image)
    #print("Image originale:")
    #image.show()
    print("hey")
    #for level in compression_level:
    #    compressed_filename = os.path.splitext(filename)[0] + f"_compressed_{level}.jpg"
    #    compress_image(image_matrix, compressed_filename, level)
    #    compressed_image = Image.open(compressed_filename)
    #    compressed_image.show()
    
    #show multiple images in a single squared figure, row and column
    fig, axes = plt.subplots(1, len(compression_level), figsize=(18, 6))
    for i, level in enumerate(compression_level):
        compressed_filename = os.path.splitext(filename)[0] + f"_compressed_{level}.jpg"
        compress_image(image_matrix, compressed_filename, level)
        compressed_image = Image.open(compressed_filename)
        axes[i].imshow(compressed_image)
        axes[i].set_title(f"Compression level: {level}")
        axes[i].axis("off")
    fig.suptitle("Compression de Jean-François Remacle avec différents niveaux")
    plt.savefig("images/compression_example.pdf")
    plt.tight_layout()
    plt.show()

compress_and_display("images/Jean-Francois_Remacle.jpg", [1, 50, 100, 300, 500])

