import urllib.request as urllib2
import certifi
import ssl
import os

# start batch / end batch
startId = 3578
endId = 3584

# Index of files for each batch
# Example: https://pds-imaging.jpl.nasa.gov/data/mro/mars_reconnaissance_orbiter/ctx/mrox_3578/index/index.tab
RequestUrL = "https://pds-imaging.jpl.nasa.gov/data/mro/mars_reconnaissance_orbiter/ctx/mrox_"
indexURL = "/index/index.tab"

for i in range(startId, endId):
    # create new path if needed
    PDS3_path = "data/CTX/" + str(i) + "/PDS3"
    if not os.path.exists(PDS3_path):
        os.makedirs(PDS3_path)
    # download index file
    indexRes = urllib2.urlopen(RequestUrL + str(i) + indexURL, context=ssl.create_default_context(cafile=certifi.where()))
    indexArray = indexRes.read().splitlines()
    dataItems = []
    # read index file
    for d in range(len(indexArray)):
        item = str(indexArray[d]).split(",")
        dataName = item[3].replace('"', '')
        dataURL = RequestUrL + str(i) + "/data/" + dataName

        print(i, d, " File:", dataName, dataURL)
        
        path = PDS3_path + "/" + str(dataName) + '.IMG'

        # skip file if image already exists, else download file and save it
        if os.path.isfile(path):
            print("Skipping: ", dataName, " file already exists")
        else:
            imgFile = urllib2.urlopen(dataURL , context=ssl.create_default_context(cafile=certifi.where()))
            imgFileBinary = imgFile.read()
            binaryFile = open(path, 'wb')
            binaryFile.write(imgFileBinary)
            binaryFile.close()