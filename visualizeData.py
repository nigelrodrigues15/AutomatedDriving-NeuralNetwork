import numpy as np
import matplotlib.pyplot as plt  # plot graphs

yVectors = (y_train, y_test, y_val)

yPlot = np.concatenate(yVectors)
xPlot = range(len(yPlot))
print(np.amin(yPlot))

plt.figure(figsize=(20, 8))
plt.xlim(0, len(yPlot))
plt.title('Data Distribution', fontsize=17)
plt.xlabel('Frames')
plt.ylabel('Steering Angle')
plt.plot(xPlot, yPlot, 'g', linewidth=0.4)

plt.show()


plt.figure(figsize=(8, 8))
plt.hist(yPlot, bins=20, color='blue', linewidth=0.1)
plt.title('Angle Histogram', fontsize=17)
plt.xlabel('Steering Angle')
plt.ylabel('Counts')

plt.show()
