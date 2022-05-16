import cv2 as cv
import numpy as np
# Load the model
net = cv.dnn.readNet("./model/model_DAD_3_7.xml",
                     "./model/model_DAD_3_7.bin")
# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# Read an image
frame = cv.imread('./model/phone_interact.jpg')
frame = cv.resize(frame, (224, 224))
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
frame = np.repeat(frame[..., np.newaxis], 3, -1)
image = np.expand_dims(frame, axis=0)

# Set Input Tensor
# blob = cv.dnn.blobFromImage(frame, ddepth=cv.CV_8U)
# print(blob.shape)
net.setInput(image)

# Perform an inference
out = net.forward()
# Out
print(out.reshape(-1, 6))
