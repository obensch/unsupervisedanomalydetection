import pickle
import numpy as np
import matplotlib.pyplot as plt

# select patch
id = 12
# patch size
patchSize = 64

patchString = "_" + str(patchSize)
patch_path = "data/patches" + patchString + "/"
patch_his_path = "data/patches_his" + patchString + "/"
patch_Canny_path = "data/patches_Canny" + patchString + "/"

img = np.load(patch_path + str(id) + ".npy")

his = np.load(patch_his_path + str(id) + ".npy")
canny = np.load(patch_Canny_path + str(id) + ".npy")

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original')
axs[0].axis('off')
axs[1].imshow(his, cmap='gray')
axs[1].set_title('Histogram Equalization')
axs[1].axis('off')
axs[2].imshow(canny, cmap='gray')
axs[2].set_title('Canny Filter')
axs[2].axis('off')
plt.show()