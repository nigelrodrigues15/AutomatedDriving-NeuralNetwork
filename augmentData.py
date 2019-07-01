
from keras.applications.inception_v3 import preprocess_input

# Define and configure ImageDataGenerator for training data.
train_datagen = ImageDataGenerator(
    zca_whitening=True)

# Define and configure ImageDataGenerator for validation data.
val_datagen = ImageDataGenerator()

'''
# Code for displaying the augemented image from train_datagen.

# Directory to save augmented images.
#aug_imgs = "drive/My Drive/cs680_aug_images"


for X_batch, y_batch in train_datagen.flow(X_train, y_train,
                                           batch_size = 9):
  
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(85, 320, 3))
    
	# show the plot
	plt.show()
	break
'''

# Define the batch size by calling the flow function.
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
