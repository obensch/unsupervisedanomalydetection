import glob
import numpy as np
import os
import time

# start batch, end batch
start_Id = 3578
end_Id = 3584

# patch size
patchSize = 64


start_time = time.time()

patchString = "_" + str(patchSize)
Patch_path = "data/patches" + patchString + "/"
Patch_meta_path = "data/patches_meta" + patchString + "/"

# create paths if needed
if not os.path.exists(Patch_path):
    os.makedirs(Patch_path)
if not os.path.exists(Patch_meta_path):
    os.makedirs(Patch_meta_path)

# Number of patches
patch_counter = 0
for id in range(start_Id, end_Id):
    NPY_path = "data/CTX/" + str(id) + "/NPY/"
    files_npy = glob.glob(NPY_path + "*.npy")
    # iterate through images
    for i in range(len(files_npy)):
        patch_meta = []
        print("processing: ", id, " number: ", i, " current patches: ", patch_counter, " file: ", files_npy[i])
        img = np.load(NPY_path + str(i) + ".npy")
        trimmed = img[:,60:-60]
        h, w = trimmed.shape
        x = 0
        y = 0
        # next line if it fits
        while (y+patchSize) < h:
            # next row if it fits
            while (x+patchSize) < w:
                patch = trimmed[y:y+patchSize,x:x+patchSize]
                patchInfo = {
                    "batchId": id,
                    "imageId": i,
                    "patchId": patch_counter,
                    "x": x,
                    "y": y,
                }  
                patch_meta.append(patchInfo)
                # save patch and increase patch counter
                np.save(Patch_path + str(patch_counter) + ".npy", patch)
                patch_counter = patch_counter + 1
                x = x + patchSize
            y = y + patchSize
            x = 0
        del img
        del trimmed
    spend = time.time() - start_time
    print("saving: ", id, " time:", spend)
    # save meta info
    np.save(Patch_meta_path + str(id) + ".npy", patch_meta)

print("--- %s seconds ---" % (time.time() - start_time))