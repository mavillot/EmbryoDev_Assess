import tensorflow_hub as hub
from DataLoaderClass import Video_Data
import keras
import tensorflow as tf


model_weights='saved_models/final_model_preTrained'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_weights,
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True),


    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
]

excel_file='Excel/ClassLabels.xlsx'
video_files='Data/Set1'
sheet_name='Set1'
data=Video_Data(video_files,excel_file,sheet_name)
train_x,train_y=data.load_data()



model_path = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
model = tf.keras.Sequential([
    hub.KerasLayer(model_path, input_shape=(None,224,224,3),trainable=False)
])

model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Softmax())

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['acc'])

NUM_EPOCHS = 100

history = model.fit(train_x,train_y,validation_split=0.3, epochs=100,callbacks=callbacks)