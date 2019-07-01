import pandas as pd

num_bins = 20

interval = 2.0/num_bins

inc = -1
bins = []
while inc <= 1:
  inc = round(inc, 1)
  bins.append(inc)
  inc += interval

### Test code

# print(bins)
# labels = np.arange(0,len(bins) - 1)
# print(labels)


# yTest = y_train[0:10]

# print('before',yTest)

# ybinned = pd.cut(yTest, bins = bins, labels = labels, include_lowest = True)

# print('after',ybinned)


####
xTrainLenth, a, b, c = X_train.shape
xTestLenth, a, b, c = X_test.shape
xValLenth, a, b, c = X_val.shape


labels = np.arange(0, len(bins) - 1)

yAll = yPlot


yBinned = pd.cut(yAll, bins=bins, labels=labels, include_lowest=True)


print(xTrainLenth)
print(xTestLenth)
print(xValLenth)
print('\n')

y_train = yBinned[0:xTrainLenth]
print(y_train.shape)

y_test = yBinned[xTrainLenth:xTrainLenth + xTestLenth]
print(y_test.shape)

y_val = yBinned[xTrainLenth + xTestLenth:xTrainLenth + xTestLenth + xValLenth]
print(y_val.shape)


print(len(y_train) + len(y_test) + len(y_val))

numberOfClasses = 20


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)

    y = np.eye(nb_classes)[targets]

    return y

# indices_to_one_hot(ytest,nb_classes)


y_train = [y_train.tolist()]
y_train = indices_to_one_hot(y_train, numberOfClasses)


y_test = [y_test.tolist()]
y_test = indices_to_one_hot(y_test, numberOfClasses)


y_val = [y_val.tolist()]
y_val = indices_to_one_hot(y_val, numberOfClasses)
