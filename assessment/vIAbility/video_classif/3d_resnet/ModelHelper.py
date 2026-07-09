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