# Writeup for Semantic Segmentation Project

_ Udacity Robotics Nanodegree Term 1 _

_ Project submitted by Ulrich Ludmann 2017/12/05 _

## Abstract
The topic of the final project of term 1 in the Udacity robotics nanodegree is about building a fully convolutional Network.
The trained network is loaded into a perception pipeline of a drone and its goal is to distinguish between "normal pedestrians" and a girl with red hair and a red shirt.
After the girl is detected, the drone should be able to follow her. There are many pedestrians crossing the targets way and sometimes the lighting changes. This ends up in a quiet complex environment.
The project assignment is
1. Acquire data out of a simulator
1. Design a neural network
1. Train the neural network with the acquired data.

The following parts describe how i addressed different assignments and in which problems I ran into.

## Acquiring Data out of a simulator
I decided not to use the given data and started to acquire my own dataset. This turned out to be a bad idea, because i ran into overfitting issues that were caused by my small dataset. In the beginning I had no Idea about how large such a dataset needs to be. Afterwards I know, that the required size of a Dataset depends on many factors.
1. The Dataset needs to cover all possible states:
e. g.:
- Pedestrians standing on a street (gray background)
- Pedestrians standing on the graas (green background)
- different camera angles (altitudes of the drone)
2. More data is better
3. Different Situations of the drone need to be covered:
- Searching for target
- following the target

So one needs to train the network on all those different situations to obtain a robust recognition.

For example My network tend to classify brighter pixel areas wrong. One possibility why this happens is because the Dataset contains more images where people are walking on the street (which has a bright pixel areas).

[example1: bright pixel areas not correctly classified]

I mixed the provided Dataset and my own dataset. In total I ended up having a big Dataset
```python
n_train_data = 17007
n_validation_data = 3999
```

So it turned out i had two major Problems:
First, my network layer size was way to big (at least for my small dataset...)


## My Semantic Segmentation Network
The network that worked best for me was a real basic network. With 3 encoder blocks, a 1x1 convolutions in the middle and 3 decoder blocks.

```python
def fcn_model(inputs, num_classes):
    layer_1 = encoder_block(inputs, 32, 2)
    layer_2 = encoder_block(layer_1, 64, 2)

    layer_1x1 = conv2d_batchnorm(layer_2, 256, kernel_size=1, strides=1)

    dec_1 = decoder_block(layer_1x1, layer_1, 64)
    dec_2 = decoder_block(dec_1, inputs, 32)
    x = dec_2

    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

[Network architecture]
In the next chapter i'm going to introduce the different types of layers.
### Layer Explanation
layer1 to layer3: Encoder layers

#### Encoder Layers
```python
def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

The encoder block takes 3 arguments. The input layer, the filter size and the strides parameter.

A convolutional filter looks at a part of an image (not on the whole image). In my code its a 3x3 kernel size so the filter looks at a 3x3 patch of the image. This is definde by the `SeperableConv2DKeras`-function:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer
```

The filter size can also be called as depth of the convolution filter. It defines of "how often" the filter looks at the patch. In my case a biggger filter size (5x5) led to longer training times and overfitting. I suppose that complex datasets needs deeper filters than our simple dataset with only 3 categories (pedestrian, target, noise).

So my encoder network is made up 3 2D-Convolutions with a 3x3 kernel size. Followed by a batch normalization to help the network in the training process.

The Convolutions have a stride of 2 which means that the output size (width x height) of the layer is half as big as the input layer.

I decrease the width and height in each network layer by 2 and increase the filter size.

### 1x1 convolutions
A 1x1 convolution adds feature maps to the decoded layers. So we can look at the whole layer as one and obtain spatial information. So if we have a 256 depth of the 1x1 filter we can have 256 different feature maps, looking at the previous layer with information on the shape of things.

A fully connected network would loose all spatial information because all neurons are connected to each others. A 1x1 convolution network connects "slices (widthxheight) of neurons".

### Decoder
The Decoder layers help us to return to the shape (width, height, depth) of the input layer. By reversing the convolution operation, we can obtain a pixel map with the same size as the input. This pixel map then contains the semantic segmentation of the image.

The Decoder block consists of a upsamling, a optional concatenation for skip layers and 2 2D-Convolution Layers that have batch normalization integrated.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):

    upsampled_layer = bilinear_upsample(small_ip_layer)

    if large_ip_layer is not None:
        concat_layers = layers.concatenate([upsampled_layer, large_ip_layer])
        x = concat_layers

    x = upsampled_layer
    output_layer = separable_conv2d_batchnorm(x, filters, strides=1)
    output_layer = separable_conv2d_batchnorm(output_layer, filters, strides=1)

    return output_layer
```

Bilinear upsampling upsamples the width and height of the input by a factor of 2.
```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```
We loose information in this procedure because as the image gets bigger we need to interpolate the pixels. In our case we used bilinear upsampling as technique. There are several other techniques e.g. remembering the pixels location from the pooling operation of the encoder. (Bed of nails upsampling http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf, page 26)


### Chosen Parameters


I chose the following parameters:
```python
learning_rate = 0.001
batch_size = 32
num_epochs = 20
n_train_data = 17007
n_validation_data = 3999
steps_per_epoch = 200 #int(n_train_data/batch_size)
validation_steps = 100# int(n_validation_data/batch_size)
workers = 2
```

With the epoch number set to 20 I was able to train the network to a IoU score of 0.385.
I than trained the network for 2 epochs on a dataset, that It has never seen before and was able to increase the IoU performance to 0.402.

I tuned the hyper parameters by brute forcing. I tried learning rates from 0.1 down to 0.0001.
Batch size is also crucial for the networks performance: if it is to small it takes long for one epoch to run. If its to high, we show the network the same dataset again and again and it tends to overfit.
I found 200 steps per epoch a good number. (Brute force)

### Limitations of the Network
As I trained the network many times to achieve the goal of IoU score of 0.40, the sample validation data bled into the training data. But the use-case of recognizing the target and following the target is working really well.

At the moment, the Network is not able to recognize other object like a dog or a cat or a car but would require an new training and as it is really simple, the number of classes it is able to distinguish is low compared to bigger networks.


## Future Enhancements
- use a pretrained network for the encoder.
- Adding more training data
-
