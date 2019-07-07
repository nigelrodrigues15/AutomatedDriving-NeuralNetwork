# Take raw data from the training simulator and convert it to a numpy array.
import os
import sys
# Importing Libraries
import csv
import cv2
# Convert a python object (list, dict, etc.) into a character stream
import numpy as np  # Matrix lib
import matplotlib
# Saving CNN weights, It lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.
import matplotlib.pyplot as plt  # plot graphs
import matplotlib.image as img  # read images
# Breaks dataset into training, validation and testing sets
from sklearn.model_selection import train_test_split
# from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
# from skimage.transform import rescale, resize, downscale_local_mean
os.chdir(r'/mnt/c/users/nigel/Desktop')
print(os.path.abspath('driving_log.csv'))
# Locations of folders in directory.
logName = os.path.abspath('driving_log.csv')
dirName = os.path.abspath('')


# Convert CSV to set of variables.
imgSet = []
steering, throttle = [], []

with open(logName) as csv_file:
    csv_log = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_log:
        if line_count == 0:
            line_count += 1
        else:
            imgSet.append(dirName + "/" + row[0].lstrip())
            imgSet.append(dirName + "/" + row[1].lstrip())
            imgSet.append(dirName + "/" + row[2].lstrip())
            
            steering.append(float(row[3]))
            steering.append(float(row[3]) + 0.2)
            steering.append(float(row[3]) - 0.2)

            line_count += 1


# fig = plt.figure()
# endCounter = 100
# crop = img.imread(imgSet[1])
# fig = plt.figure(figsize = (10,10))
# plt.imshow(crop)
# plt.show()

print(imgSet[1])

# Converting images to numpy arrays
width, height = 320, 85

# labels, Y
Y = steering
Y_flip = [n * -1 for n in Y]
# print(len(Y_flip))

# Input Data, X
X = np.zeros([2*len(Y), height, width, 3], dtype=np.float32)
# X_flip = np.zeros([len(Y_flip), height, width], dtype=np.float32)

counter = 0
# Populating X matrix
for imgName in imgSet:
    if counter%100 == 0:
        print(counter)
#     if counter > endCounter:
#         break
    image = img.imread(imgName)
    
    # image = rgb2gray(image)

    # Normalize values
    X[counter,:,:] = image[60:145,:,:]/255.0
    X[counter + len(Y), :, :] = cv2.flip(X[counter, :, :], 1)

    counter += 1

    
# Append steering angles and form a nx1 vector.
Y_new = Y + Y_flip

# Y array.

Y = np.array(Y_new)


# Split the dataset into training, validation, and testing subsets.
X_half1, X_half2, y_half1, y_half2 = train_test_split(X, Y, test_size=0.50, random_state=99)

X_train, X_test, y_train, y_test = train_test_split(X_half1, y_half1, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.50, random_state=30)


# Save imported data.
np.save(dirName + '/X_train.npy', X_train)
np.save(dirName + '/y_train.npy', y_train)

np.save(dirName + '/X_test.npy', X_test)
np.save(dirName + '/y_test.npy' , y_test)

np.save(dirName + '/X_val.npy' , X_val)
np.save(dirName + '/y_val.npy' , y_val)

print("Numpy arrays saved :)")


