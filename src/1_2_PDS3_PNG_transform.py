import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# start batch, end batch
startId = 3578
endId = 3584


for id in range(startId, endId):
    print("Converting batch: ", id)
    start_time = time.time()
    PDS_path = "CTX/" + str(id) + "/PDS3/"
    NPY_path = "CTX/" + str(id) + "/NPY_PNG/"
    PNG_path = "CTX/" + str(id) + "/PNG/"

    # create paths if needed
    if not os.path.exists(NPY_path):
        os.makedirs(NPY_path)
    
    if not os.path.exists(PNG_path):
        os.makedirs(PNG_path)

    mapping = []
    # ! convert images with pds3 transform library, needs to be installed (java) !
    files = glob.glob(PDS_path + "*.img")
    for img in range(len(files)):
        print("Transforming: ", img , " File:", files[img])
        cmd_transform = 'transform ' + files[img] + " -f png -o " + PNG_path
        os.system(cmd_transform)

    # convert images to numpy arrays 
    files_PNG = glob.glob(PNG_path + "*.png")
    for img in range(len(files_PNG)):
        print("Converting to numpy: ", img , " File:", files_PNG[img])
        image = cv2.imread(files_PNG[img], cv2.IMREAD_GRAYSCALE)
        npy = image[:,:]
        # save numpy array and meta info
        print("Saving: ", img , " File:", files_PNG[img])
        np.save(NPY_path + str(img) + ".npy", npy)
        meta = {
            "id": img,
            "name": os.path.basename(files_PNG[img]),
        }
        mapping.append(meta)
    np.save("CTX/" + str(id) + "/mapping_PNG.npy", mapping)

    print("Batch: ", id, " converted in: ", str(time.time() - start_time), " seconds" )