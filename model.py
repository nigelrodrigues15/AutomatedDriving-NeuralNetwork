import tensorflow
from tensorflow import keras
from keras.models import model_from_yaml
from keras import optimizers

import numpy as np

def main()

    # Imported npy data files.
    X_train = np.load('C:\\Users\\nigel\\Documents\\Coding_Directory\\AutomatedDriving-NeuralNetwork\\ProcessedData\\X_train.npy')
    y_train = np.load('C:\\Users\\nigel\\Documents\\Coding_Directory\\AutomatedDriving-NeuralNetwork\\ProcessedData\\y_train_20.npy')

    X_test = np.load('C:\\Users\\nigel\\Documents\\Coding_Directory\\AutomatedDriving-NeuralNetwork\\ProcessedData\\X_test.npy')
    y_test = np.load('C:\\Users\\nigel\\Documents\\Coding_Directory\\AutomatedDriving-NeuralNetwork\\ProcessedData\\y_test_20.npy')

    X_val = np.load('C:\\Users\\nigel\\Documents\\Coding_Directory\\AutomatedDriving-NeuralNetwork\\ProcessedData\\X_val.npy')
    y_val = np.load('C:\\Users\\nigel\\Documents\\Coding_Directory\\AutomatedDriving-NeuralNetwork\\ProcessedData\\y_val_20.npy')

    print("Numpy data imported :)")


    # load YAML and create model
    yaml_file = open('G:\\My Drive\\ME780AutomatedDriving\\trainingDataset\\LeNet\\model_net.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model.load_weights('G:\\My Drive\\ME780AutomatedDriving\\trainingDataset\\LeNet\\model_net.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                        loss='mean_squared_error',
                        metrics=['acc'])
    score = loaded_model.evaluate(X_train, y_train, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

if __name__ == "__main__":
    main()
    pass