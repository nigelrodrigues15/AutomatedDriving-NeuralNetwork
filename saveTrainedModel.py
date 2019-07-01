# evaluate the model
scores = model_net.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model_net.metrics_names[1], scores[1]*100))

with open(dirName + '/LeNet/history', 'wb') as file_pi:
  pickle.dump(history_net.history, file_pi)

# Serialize model to YAML
model_net_yaml = model_net.to_yaml()

with open(dirName + '/LeNet/model_net.yaml', 'w') as yaml_file:
    yaml_file.write(model_net_yaml)

# serialize weights to HDF5
model_net.save_weights(dirName + '/LeNet/model_net.h5')
print("Saved model to disk")
