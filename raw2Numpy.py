# Take raw data from the training simulator and convert it to a numpy array

# Importing Libraries

import csv
# convert a python object (list, dict, etc.) into a character stream
import numpy as np  # Matrix lib
# saving CNN weights, It lets you store huge amounts of numerical data, and easily manipulate that data from NumPy.
import matplotlib.pyplot as plt  # plot graphs
import matplotlib.image as img  # read images
# breaks dataset into training, validation and testing sets
from sklearn.model_selection import train_test_split

# Locations of folders in directory
logName = "G:\\My Drive\\ME780AutomatedDriving\\driving_log.csv"
dirName = "G:\\My Drive\\ME780AutomatedDriving\\trainingDataset"

# Convert CSV to set of variables

imgSet = []
steering, throttle, brake, speed = [], [], [], []

with open(logName) as csv_file:
    csv_log = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_log:
        if line_count == 0:
            line_count += 1
        else:
            imgSet.append(dirName + "\\" + row[0].lstrip())
            imgSet.append(dirName + "\\" + row[1].lstrip())
            imgSet.append(dirName + "\\" + row[2].lstrip())
            for i in range(0, 3):
              steering.append(float(row[3]))
              throttle.append(float(row[4]))
              brake.append(float(row[5]))
              speed.append(float(row[6]))

            line_count += 1


# fig = plt.figure()
# # endCounter = 29
# crop = img.imread(imgSet[1])
# # fig = plt.figure(figsize = (10,10))
# plt.imshow(crop)
# plt.show()

print(imgSet[1])

# Converting images to numpy arrays
width, height = 85, 320

# labels, Y
Y = steering

# Input Data, X
X = np.zeros([len(Y), width, height, 3], dtype=np.float32)

counter = 0
# Populating X matrix

for imgName in imgSet:
#   if counter > endCounter:
#     break

  image = img.imread(imgName)

  # Crop and Normalize values
  X[counter] = image[60:145, :, :]/255.0

  counter += 1

print(X.shape)


# Split the dataset into training, validation, and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.50, random_state=30)

plt.imshow(X_train[10,:,:,:])
plt.show()

# Save imported data.
np.save('C:\\Users\\nigel\\Desktop\\ProcessedData\\X_train.npy', X_train)
np.save('C:\\Users\\nigel\\Desktop\\ProcessedData\\y_train.npy', y_train)

np.save('C:\\Users\\nigel\\Desktop\\ProcessedData\\X_test.npy', X_test)
np.save('C:\\Users\\nigel\\Desktop\\ProcessedData\\y_test.npy' , y_test)

np.save('C:\\Users\\nigel\\Desktop\\ProcessedData\\X_val.npy' , X_val)
np.save('C:\\Users\\nigel\\Desktop\\ProcessedData\\y_val.npy' , y_val)

print("Numpy arrays saved :)")
