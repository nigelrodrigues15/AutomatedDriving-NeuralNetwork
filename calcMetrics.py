# Extract accuracy and loss for plotting.
acc_net = history_net.history['acc']
val_acc_net = history_net.history['val_acc']
loss_net = history_net.history['loss']
val_loss_net = history_net.history['val_loss']

epochs = range(1, len(acc_net) + 1)

plt.figure()
plt.plot(epochs, acc_net, 'm', label='Training Accuracy')
plt.plot(epochs, val_acc_net, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy for LeNet-5 \n Network', size=20)
plt.xlabel('Number of Epochs', size=20)
plt.ylabel('Accuracy (Fraction)', size=20)
plt.legend()

plt.figure()
plt.plot(epochs, loss_net, 'm', label='Training Loss')
plt.plot(epochs, val_loss_net, 'b', label='Validation Loss')
plt.title('Training and Validation Loss for LeNet-5 \n Network', size=20)
plt.xlabel('Number of Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend()

plt.show()
