import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import accuracy_score


def create_model1():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def create_model2(conv_size, conv_depth):
    input = layers.Input(shape=(28, 28, 1))

    x = input
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, 'relu')(x)
    x = layers.Dense(10, 'sigmoid')(x)
    model = models.Model(inputs=input, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs, destination_file):
    model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2)])
    model.save(destination_file + '.h5')


def test_model(model, x_test, y_test):
    accuracy = model.evaluate(x_test, y_test, verbose=1)[1]
    return accuracy
