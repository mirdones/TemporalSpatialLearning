import os
import time
from datetime import timedelta

import numpy as np
import tqdm as tqdm
from tensorflow import keras

from sklearn.metrics import classification_report, confusion_matrix

from DataGenerator import DataGenerator, TRAIN, VALIDATION, ALL
from Model import getModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# model_name = 'left_right'

num_epochs = 1000

freedom = 8

if freedom == 2:
    model_name = 'left_right'
elif freedom == 8:
    model_name = '8_directions'

patience = 10

train_generator = DataGenerator(mode=TRAIN, freedom=freedom)
val_generator = DataGenerator(mode=VALIDATION, freedom=freedom)
all_generator = DataGenerator(mode=ALL, freedom=freedom)

model = getModel(decay_steps=5 * len(train_generator), categories=freedom,
                 initial_learning_rate=1e-3, model_name=model_name)

tb_callback = keras.callbacks.TensorBoard(log_dir=f'./logs/{model_name}', histogram_freq=1, write_graph=False,
                                          write_images=False)

early_stop_loss = keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss', restore_best_weights=True)
early_stop_acc = keras.callbacks.EarlyStopping(patience=patience, monitor='val_accuracy', restore_best_weights=True)

print(f"--- Training {model_name} ---\n")

start_time = time.monotonic()

history = model.fit(
    # x=x,
    # y=y_true,
    # validation_split=0.2,
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs,
    callbacks=[tb_callback, early_stop_loss,
               # early_stop_acc
               ])

end_time = time.monotonic()
executionTime = timedelta(seconds=end_time - start_time)

print(f"--- {executionTime} to train {model_name} ---\n")

print(f"\n--- Evaluating {model_name} ---\n")

start_time = time.monotonic()

loss, acc = model.evaluate(all_generator, verbose=2)

end_time = time.monotonic()
executionTime = timedelta(seconds=end_time - start_time)

print(f"--- {executionTime} to evaluate {model_name} ---\n")

print(f"\n--- Getting classification report {model_name} ---\n")

start_time = time.monotonic()

x = []
y_true = []
y_pred = []
for i in tqdm.tqdm(range(len(all_generator))):
    batch_X, batch_y_true = all_generator.__getitem__(i)
    batch_size = len(batch_y_true)
    batch_y_pred = model.predict(batch_X, verbose=0)
    x.extend(batch_X)
    y_true.extend(batch_y_true)
    y_pred.extend(batch_y_pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Convert true labels to class indices
y_true_indices = np.argmax(y_true, axis=1)

# Convert predicted values to class indices
y_pred_indices = np.argmax(y_pred, axis=1)

# Compute multilabel confusion matrix
confusion_matrix = confusion_matrix(y_true_indices, y_pred_indices)
print(f'\n{confusion_matrix=}\n')

# Compute classification report
report = classification_report(y_true_indices, y_pred_indices)
print(f'\n{report=}\n')

end_time = time.monotonic()
executionTime = timedelta(seconds=end_time - start_time)

print(f"--- {executionTime} to get classification report from {model_name} ---\n")

# Step 6: Print and save the confusion matrix
cm_output = "Confusion Matrix:\n" + str(confusion_matrix)
print(cm_output)
with open('confusion_matrix.txt', 'w') as f:
    print(cm_output, file=f)

# Step 7: Print and save the classification report
report_output = "Classification Report:\n" + report
print(report_output)
with open('classification_report.txt', 'w') as f:
    print(report_output, file=f)

model.save(f'{model_name}.h5')
