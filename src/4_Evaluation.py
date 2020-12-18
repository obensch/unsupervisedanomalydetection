import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import cv2 

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# settings
startId = 0
endId = 44
patchSize = 64
batchId = 3579
his = True
canny = False
model = "64_LS_1mT_200kV_L512_E3_C32.md"

model_path = "data/models/"
NPY_path = "data/CTX/" + str(batchId) + "/NPY/"

model = load_model(model_path + model)

anomalies = []
for n in range(startId, endId):
    img = np.load(NPY_path + str(n) + ".npy")
    trimmed = img[:,60:-60]
    h, w = trimmed.shape

    x = int(w/patchSize)
    y = int(h/patchSize)
    y_max = y*patchSize
    x_max = x*patchSize

    target = trimmed[:y_max,:x_max]

    targetImg = np.zeros((y_max, x_max))
    resultImg = np.zeros((y_max, x_max))
    resAnomaly = np.zeros((y_max, x_max))

    patches = np.zeros((x*y, patchSize, patchSize))
    targetPatches = np.zeros((x*y, patchSize, patchSize))

    i = 0
    for yp in range(y):
        for xp in range(x):
            y_start = yp*patchSize
            y_end = (yp+1)*patchSize
            x_start = xp*patchSize
            x_end = (xp+1)*patchSize

            patch = target[y_start:y_end, x_start:x_end]
            if his:
                patch = cv2.equalizeHist(patch)
            if canny:
                patch = cv2.Canny(patch, 10, 20)
            patches[i] = patch
            targetPatches[i] = patch
            i = i + 1

    patches = patches.astype('float32') / 255
    patches.reshape(x*y, patchSize, patchSize, 1)

    res_patches = model.predict(patches)

    # patch = (res_patches[0]* 255).astype("uint8")

    i = 0
    for yp in range(y):
        for xp in range(x):
            y_start = yp*patchSize
            y_end = (yp+1)*patchSize
            x_start = xp*patchSize
            x_end = (xp+1)*patchSize
            mse = ((res_patches[i] - patches[i])**2).mean()
            patch = (res_patches[i]* 255).astype("uint8")
            resultImg[y_start:y_end, x_start:x_end] = patch[:,:,0]
            targetImg[y_start:y_end, x_start:x_end] = targetPatches[i]
            resAnomaly[y_start:y_end, x_start:x_end] = mse
            i = i+1
    print(max(resAnomaly))
    anomalies.append(resAnomaly.mean())

i=0
for n in range(startId, endId):
    print("Image:", n, " mean:", anomalies[i])
    i = i+1