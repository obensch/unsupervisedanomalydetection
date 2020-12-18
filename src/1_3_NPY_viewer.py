import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import environ
environ["OPENCV_IO_ENABLE_JASPER"] = "true"
import cv2

# select image id and batch to inspect image
id = 37
batch = 3578

# set to false if histogram equalization and canny filter should be applied
showOrginalOnly = True

NPY_path = "data/CTX/" + str(batch) + "/NPY/" + str(id) + ".npy"

img = np.load(NPY_path)
trimmed = img[:,60:-60]

# show image only
if showOrginalOnly:
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(trimmed, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    plt.show()
else: 
    # display images with histogram equalization and canny filter
    his = cv2.equalizeHist(trimmed)
    canny = cv2.Canny(trimmed, 10, 20)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(trimmed, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(his, cmap='gray')
    axs[1].set_title('Histogram')
    axs[1].axis('off')
    axs[2].imshow(canny, cmap='gray')
    axs[2].set_title('Canny')
    axs[2].axis('off')
    plt.show()