from jax import nn as jnn, numpy as jnp, random
from flax import linen as nn
from typing import Sequence

#Each tuple represents a convolutional or maxpool layer in the YoloV1 architecture
#(Features, Dimension of Kernel Size, Stride, Padding on either side) for convolutional layers
#(0,) represents 2 X 2 max pool layer with stride of 2
MODEL_ARCHITECTURE = [
  (64, 7, 2, 3),
  (0,),
  (192, 3, 1, 1),
  (0,),
  (128, 1, 1, 0),
  (256, 3, 1, 1),
  (256, 1, 1, 0),
  (512, 3, 1, 1),
  (0,),
  (256, 1, 1, 0),
  (512, 3, 1, 1),
  (256, 1, 1, 0),
  (512, 3, 1, 1),
  (256, 1, 1, 0),
  (512, 3, 1, 1),
  (256, 1, 1, 0),
  (512, 3, 1, 1),
  (512, 1, 1, 0),
  (1024, 3, 1, 1),
  (0,),
  (512, 1, 1, 0),
  (1024, 3, 1, 1),
  (512, 1, 1, 0),
  (1024, 3, 1, 1),
  (1024, 3, 2, 1),
  (1024, 3, 1, 1),
  (1024, 3, 1, 1)
]

#Each max pool layer in YoloV1 is identical
def max_pool_layer(x):
  return nn.max_pool(x, (2, 2), (2, 2))

class YoloV1(nn.Module):
  #Properties of each convolutional layer
  conv_structures: Sequence[tuple]

  #split size
  S: int

  #number of bounding boxes per grid position
  B: int

  #number of classes
  C: int

  #For N X N output grid, split_size represents N
  #num_boxes represents number of bounding boxes per grid position
  #num_classes represents how many object classes the model can detect
  #split_size X split_size is number of grid positions
  #num_boxes * (x, y, w, h, and confidence for each box) + one probability value for each class per box is number of values per grid position
  #multiply values per grid spot by number of grid spots for number of neurons in output layer
  def get_output_length(self, split_size, num_boxes, num_classes):
    return split_size * split_size * (5 * num_boxes + num_classes)

  def setup(self):
    #converting the model architecture to flax layers
    self.conv_layers = [nn.Conv(conv_structure[0], (conv_structure[1], conv_structure[1]), (conv_structure[2], conv_structure[2]), [(conv_structure[3], conv_structure[3]), (conv_structure[3], conv_structure[3])]) if len(conv_structure)==4 else max_pool_layer for conv_structure in self.conv_structures]
    
    #actual model has hidden layer with 4096 neurons, using 496 to make training/inference time more reasonable
    self.dense_layers = [nn.Dense(496), nn.Dense(self.get_output_length(self.S, self.B, self.C))]

  def __call__(self, inputs):
    x = inputs
    for conv_layer in self.conv_layers:
      x = conv_layer(x)
      print(x.shape) #make sure the shapes of each layer match the paper's model architecture
      
      #activation function for each convolutional layer
      if conv_layer != max_pool_layer:
        x = jnn.leaky_relu(x, 0.1)
    
    #flattening to pass into dense layers
    x = jnp.ravel(x)

    for i, dense_layer in enumerate(self.dense_layers):
      x = dense_layer(x)
      if i != len(self.dense_layers) - 1:
        x = jnn.leaky_relu(x)
    return x