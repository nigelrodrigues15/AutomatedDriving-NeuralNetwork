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
import pandas as pd
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
width, height = 320, 80

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
    if counter % 100 == 0:
        print(counter)
#     if counter > endCounter:
#         break
    image = img.imread(imgName)

    # image = rgb2gray(image)

    # Normalize values
    X[counter, :, :] = image[60:140, :, :]/255.0
    X[counter + len(Y), :, :] = cv2.flip(X[counter, :, :], 1)

    counter += 1


# Append steering angles and form a nx1 vector.
Y_new = Y + Y_flip

# Y array.

Y = Y_new

mode_1 = max(set(Y), key=Y.count)
print(mode_1)

idx1 = []

for index, val in enumerate(Y):
    if (val > mode_1-0.01 and val < mode_1+0.01):
        idx1.append(index)

idx1.reverse()

for i in idx1[0:2000]:
  del Y[i]

mode_2 = max(set(Y), key=Y.count)
print(mode_2)

idx2 = []

for index, val in enumerate(Y):
    if (val > mode_2-0.01 and val < mode_2+0.01):
        idx2.append(index)

idx2.reverse()

for i in idx2[0:2000]:
  del Y[i]

mode_3 = max(set(Y), key=Y.count)
print(mode_3)

idx3 = []

for index, val in enumerate(Y):
    if (val > mode_3-0.01 and val < mode_3+0.01):
        idx3.append(index)

idx3.reverse()

for i in idx3[0:2000]:
  del Y[i]

Y = np.array(Y)


# Plot of accuracies as a function of contributing neighbors.
def plot_steeringHist(steeringAngle):

    fig = plt.figure()

#     plt.figure(figsize = (8,8))
    plt.hist(steeringAngle, bins=100, color='blue', linewidth=0.5)

    plt.title('Steering Angle Histogram', fontsize=25)
    plt.xlabel('Steering Angle', size=20)
    plt.ylabel('Counts', size=20)

    plt.show()

    fig.savefig('./steeringAngleHist.png', bbox_inches='tight')

    return


plot_steeringHist(Y)


# # Split the dataset into training, validation, and testing subsets.
# X_half1, X_half2, y_half1, y_half2 = train_test_split(
#     X, Y, test_size=0.65, random_state=99)

# # Number of desired bins -> Number of labels.
# num_bins = 100

# # Corresponding intervals for the number of bins.
# interval = 2.0/num_bins

# # Initialize increment at inc equal to the lower bound and bins as an empty list
# # then append interval to inc until the upper bound is passed.
# inc = -1
# bins = []
# while inc <= 1:
#   inc = round(inc, 2)
#   bins.append(inc)
#   inc += interval

# # Label vector.
# labels = np.arange(0, len(bins) - 1)

# # Bin the y-values of yAll into yBinned.
# yAll = y_half1
# X_combined = X_half1
# yBinned = pd.cut(yAll, bins=bins, labels=labels, include_lowest=True)


# # Check for NANs.
# print("is any NaN?", np.isnan(yBinned).any())

# # Remove values that occur out the range [-1,1].
# y_outside = np.isnan(yBinned)
# y_remove = np.where(y_outside)[0]

# y_reversed = np.flip(y_remove)

# for i in y_reversed:
#   print(yBinned[i])
#   yBinned = np.delete(yBinned, i)
#   X_combined = np.delete(X_combined, i, 0)

# print("is NaN?", np.isnan(yBinned).any())
# print("Xshape", X_combined.shape)
# print("yshape", yBinned.shape)



# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, yBinned, test_size=0.30, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_test, y_test, test_size=0.50, random_state=30)

# y_train = np.array([int(i) for i in y_train])
# y_test = np.array([int(i) for i in y_test])
# y_val = np.array([int(i) for i in y_val])

# Save imported data.
# np.save(dirName + '/X_train.npy', X_train)
# np.save(dirName + '/y_train.npy', y_train)

# np.save(dirName + '/X_test.npy', X_test)
# np.save(dirName + '/y_test.npy', y_test)

# np.save(dirName + '/X_val.npy', X_val)
# np.save(dirName + '/y_val.npy', y_val)

print("Numpy arrays saved :)")
