import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path='/datasets/fruits_small/'):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    train_data = datagen.flow_from_directory(
        directory=path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345
    )

    return train_data


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(96, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=1,
               steps_per_epoch=None, validation_steps=None):

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model