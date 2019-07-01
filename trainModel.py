with tf.device('/device:GPU:0'):
    history_net = model_net.fit_generator(train_generator,
                                          steps_per_epoch=len(X_train)/32,
                                          validation_steps=50,
                                          validation_data=val_generator,
                                          shuffle=True,
                                          epochs=100)

