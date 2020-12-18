import glob
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time

# start batch, end batch
start_Id = 3578
end_Id = 3584

# patch size
patchSize = 64

start_time = time.time()

patchString = "_" + str(patchSize)
    
Patch_path = "data/patches" + patchString + "/"
Patch_his_path = "data/patches_Canny" + patchString + "/"

# create path if needed
if not os.path.exists(Patch_his_path):
    os.makedirs(Patch_his_path)

files_npy = glob.glob(Patch_path + "*.npy")

for i in range(len(files_npy)):
    if (i%10000) == 0:
        spend = time.time() - start_time
        print("Processing: ", i, " time: ", spend)

    # apply canny filter to each patch
    name = os.path.basename(files_npy[i])
    patch = np.load(files_npy[i])
    dst = cv2.Canny(patch, 10, 20)

    # save patch
    np.save(Patch_his_path + name, dst)

spend = time.time() - start_time
print("Done: ", i, " time: ", spend)