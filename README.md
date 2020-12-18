# Unsupervised Anomaly Detection in MRO-CTX images

## requirements

Python version: 3.8

The following python packages are required

- numpy
- openCV
- imutils
- time
- tensorflow/keras
- either "planetaryimage" (slow) or NASA PDS transform (Java, fast)(https://github.com/NASA-PDS/transform)

The source can be found in the "src" folder.
The data is saved into the "data" folder.

# Pipeline

"0_DownloadPDS3.py" can be used to download the PDS3 images into the "data/CTX/BATCHNUMBER/PDS3/" folder.

"1_1_PDS3_Planetary_convert.py" (planetaryimage) OR "1_2_PDS3_PNG_transform.py" (NASA PDS transform) can be used to convert PDS3 images to numpy (Folder: "data/CTX/BATCHNUMBER/NPY/").

"1_3_NPY_viewer.py" can be used to view the converted numpy images.

"2_1_Generate_patches.py" can be used to slice the numpy images into patches (Folder: "data/patches_PATCHSIZE"), used for training.

"2_2_Patch_histogram.py" loads image patches applies histogram equalization on them (Folder: "data/patches_his_PATCHSIZE").

"2_3_Patch_edge.py" loads image patches applies the OpenCV canny filter on them (Folder: "data/patches_Canny_PATCHSIZE").

"2_4_Patch_viewer.py" can be used to view the created patches.

"3_Model.py" can be used to train a model with the created patches. The model is saved to "data/models/SETTINGS.md".

"4_Evaluation.py" can be used to evaluate an batch of images with a selected model.

"5_Visualization.py" can be used to visualize an image (Original image, e.g. histogram equalization input, reconstruction image and MSE image).
