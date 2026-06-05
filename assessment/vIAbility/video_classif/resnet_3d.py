import einops
import keras
from keras import layers

class conv_blocks(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.layer1 = layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                    padding=padding)
        self.layer2 = layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1),
                                    padding=padding)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Project(keras.layers.Layer):
  def __init__(self, units):
    super().__init__()
    self.layer1 = layers.Dense(units)
    self.layer2 = layers.LayerNormalization()

  def call(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x


class ResidualMain(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.layer1 = conv_blocks(filters=filters, kernel_size=kernel_size, padding='same')
        self.layer2 = layers.LayerNormalization()
        self.layer3 = layers.ReLU()
        self.layer4 = conv_blocks(filters=filters, kernel_size=kernel_size, padding='same')
        self.layer5 = layers.LayerNormalization()

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos

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
