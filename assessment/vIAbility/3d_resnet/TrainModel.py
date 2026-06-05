from Model import get_model
from DataLoaderClass import Video_Data
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

model_weights_loss='saved_models/final_modelEC_loss.keras'
model_weights_acc='saved_models/final_modelEC_acc.keras'

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_weights_loss,
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True),

    tf.keras.callbacks.ModelCheckpoint(model_weights_acc,
                                       monitor='val_accuracy',
                                       mode='max',
                                       save_best_only=True),

    #tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10, mode="min"),

    #tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
]

excel_file='Excel/ClassLabels_small.xlsx'
video_files='Data/SmallSet1'
sheet_name='Set1'
data=Video_Data(video_files,excel_file,sheet_name)
train_x,train_y=data.load_data()

print(train_x.shape)
print(train_y.shape)

model=get_model()
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001), #
              metrics = ['accuracy'])

history = model.fit(train_x,train_y, batch_size=4, epochs = 150, validation_split=0.3, callbacks = callbacks)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()