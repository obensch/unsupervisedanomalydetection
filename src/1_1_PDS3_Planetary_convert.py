import glob
import os
import numpy as np
from planetaryimage import PDS3Image

# Start batch, end batch
startId = 3578
endId = 3584


for id in range(startId, endId):
    # create path if needed
    PDS_path = "CTX/" + str(id) + "/PDS3/"
    NPY_path = "CTX/" + str(id) + "/NPY/"
    if not os.path.exists(NPY_path):
        os.makedirs(NPY_path)

    # get all images
    files = glob.glob(PDS_path + "*.img")
    mapping = []
    for img in range(len(files)):
        # convert each image and save meta information
        print("Opening: ", img , " File:", files[img])
        image = PDS3Image.open(files[img])
        npy = image.image[:,:]
        print("Saving: ", img , " File:", files[img])
        np.save(NPY_path + str(img) + ".npy", npy)
        meta = {
            "id": img,
            "name": os.path.basename(files[img]),
        }
        mapping.append(meta)
    np.save("CTX/" + str(id) + "/mapping.npy", mapping)