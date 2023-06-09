import os

from keras import backend as K, Input
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def getModel(decay_steps, model_name, categories=2, initial_learning_rate=1e-3):
    input_shape = (None, 256)
    model_input = Input(shape=input_shape)

    LSTM = keras.layers.lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(model_input)

    LSTM2 = keras.layers.lstm = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False))(LSTM)

    model_output = keras.layers.Dense(categories, activation='softmax')(LSTM2)

    model = keras.models.Model(model_input, model_output)

    model.build(input_shape)

    model.summary()

    keras.utils.plot_model(model, to_file=f'{model_name}.png', expand_nested=True, show_shapes=True)

    decay_rate = 0.9

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
