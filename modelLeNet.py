# Google LeNet-5 network.
model_net = models.Sequential()

model_net.add(layers.Conv2D(filters=6, kernel_size=(3, 3),
                            activation='relu', input_shape=(85, 320, 3)))
model_net.add(layers.AveragePooling2D())

model_net.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model_net.add(layers.AveragePooling2D())

model_net.add(layers.Flatten())

model_net.add(layers.Dense(units=120, activation='relu'))

model_net.add(layers.Dense(units=84, activation='relu'))

model_net.add(layers.Dense(units=20, activation='softmax'))

model_net.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

model_net.summary()
