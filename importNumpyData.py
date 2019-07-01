import numpy as np
dirName = ""

# Imported npy data files.
X_train = np.load(dirName + '\\ProcessedData\\X_train.npy')
y_train = np.load(dirName + '\\ProcessedData\\y_train.npy')

X_test = np.load(dirName + '\\ProcessedData\\X_test.npy')
y_test = np.load(dirName + '\\ProcessedData\\y_test.npy')

X_val = np.load(dirName + '\\ProcessedData\\X_val.npy')
y_val = np.load(dirName + '\\ProcessedData\\y_val.npy')

print("Numpy data imported :)")
