from tensorflow import keras
import config as conf
from datapreparation import get_train_data, get_validation_data
import matplotlib.pyplot as plt


def create_model():
    model = keras.models.Sequential([
        keras.layers.Normalization(),
        keras.layers.Dense(units=4096, input_dim=conf.TARGET_SIZE, activation="relu"),
        keras.layers.Dense(units=4096, activation="relu"),
        keras.layers.Dense(units=2048, activation="relu"),
        keras.layers.Dense(units=2048, activation="relu"),
        keras.layers.Dense(units=1024, activation="relu"),
        keras.layers.Dense(units=512, activation="relu"),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_model(model: keras.Model):
    x, y = get_train_data()
    valid_x, valid_y = get_validation_data()

    history = model.fit(x=x, y=y, epochs=10, validation_data=(valid_x, valid_y))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.show()

    model.save("trained_model")


def create_train():
    model = create_model()
    train_model(model)


