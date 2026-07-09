import keras
from ModelHelper import conv_blocks,ResizeVideo,ResidualMain,Project
from keras import layers

def add_residual_block(input, filters, kernel_size):
  out = ResidualMain(filters, kernel_size)(input)

  res = input
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

"""Build a 3D convolutional neural network model."""
def get_model(width=224, height=224, depth=3):
    input_shape = (None, 25, height, width, depth)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = conv_blocks(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(3)(x)
    output = layers.Softmax()(x)

    # Define the model.
    model = keras.Model(input, output, name="VideoClassifier")
    #print(model.summary())
    return model


# Build model.

