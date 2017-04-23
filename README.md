# Car Behaviour Cloning
Behavior cloning of driving a car on a track. The purpose of the program is to
train a neural network to mimic driving behaviour of a human. The driving data
is collected from a simulation.

To correctly mimic human behaviour in a car driving simulator We undertook the following steps:
 1. Build a convolutional neural network using Keras in Python to predict steering angles.
 2. Train and validate the model.
 3. Test and learn the architecture.
 4. Generate more data using the simulator.

Below We will discuss each of the steps We undertook.

# Build a Convolutional Neural Network in Keras
We use a convolutional neural network with a wide residual block to predict the
steering angle. We started with the [NVidia architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 
Where our implementation is different from NVidia's is that we add a wide
residual block within the last part of the convolutional layers following
[Zagoruyko, 20016](https://arxiv.org/abs/1605.07146) as well as batch normalization
and dropout. To reduce the size of the neural network we added an average pooling
layer. The final architecture looks as follows,

![final model](images/final_model.png)

The summary per layer is as follows,

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
images (InputLayer)          (None, 3, 80, 80)         0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 3, 80, 80)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 40, 40)        2432      
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 40, 40)        160       
_________________________________________________________________
elu_1 (ELU)                  (None, 32, 40, 40)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 20, 20)        19224     
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 20, 20)        80        
_________________________________________________________________
elu_2 (ELU)                  (None, 24, 20, 20)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 36, 20, 20)        21636     
_________________________________________________________________
batch_normalization_3 (Batch (None, 36, 20, 20)        80        
_________________________________________________________________
elu_3 (ELU)                  (None, 36, 20, 20)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 36, 20, 20)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 24, 20, 20)        21624     
_________________________________________________________________
add_1 (Add)                  (None, 24, 20, 20)        0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 24, 20, 20)        80        
_________________________________________________________________
elu_4 (ELU)                  (None, 24, 20, 20)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 64, 20, 20)        13888     
_________________________________________________________________
batch_normalization_5 (Batch (None, 64, 20, 20)        80        
_________________________________________________________________
elu_5 (ELU)                  (None, 64, 20, 20)        0         
_________________________________________________________________
average_pooling2d_1 (Average (None, 64, 9, 9)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 5184)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              6035340   
_________________________________________________________________
dense_2 (Dense)              (None, 512)               596480    
_________________________________________________________________
dense_3 (Dense)              (None, 100)               51300     
_________________________________________________________________
dense_4 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 51        
_________________________________________________________________
steering_output (Dense)      (None, 1)                 2         
=================================================================
Total params: 6,767,507.0
Trainable params: 6,767,267.0
Non-trainable params: 240.0
_________________________________________________________________
```

To prevent overfitting we add L2 regularization to each layer with a threshold
of 0.001. Throughout the network we use 
[ELU activations](https://arxiv.org/pdf/1511.07289.pdf). The advantage of ELU
activation is that they converge faster and lead to higher accuracy and don't
have problems with vanishing gradient. The final mapping to the steering angle
uses a Tanh activation. 

After activation we perform Batch Normalization (Ioffe, S. and Szegedy, C. 2014)[http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf] in line with the original
residual net paper [Zagoruyko, 20016](https://arxiv.org/abs/1605.07146). The
benefit of the Batch Normalization is that the network does not need to learn
the scale or the shift in the data flowing in the layer.

We initialize the weights with a Glorot normal distribution 
[Glorot, X. and Bengo, Y. 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).

We optimize the model with the AdamX optimizer and a mean square error (MSE)
measurement. The weights are saved every time that the model finds a new lowest
MSE on the validation set. Since we want to keep a large amount of data for training
we hold out ten percent of the data for validation.


## Model input
The model input consists of images that are recorded in a car simulator. There
are recordings from the left, center and right cameras. Moreover, the behaviour
of the driver is recorded in a log file. In the log file the steering angle is
recorded as well as the throttle and the break. We initially developed
adaptations of the NVidia model where we had multiple inputs and multiple
outputs. We found that using this model to drive the car lead to poor results.
The car would immediately drive off the road. Instead we opted to use a constant
throttle and break and to only use the model to predict the steering angle.

In the first model we used the left, center and right camera as separate inputs
to the model. We found however, that the simulation feeds only a center image
when driving. As a result we could not use this model. We therefore decided to
load all the images in a single array with according steering angles.

### Cropping
To reduce the dimensionality of the calibration we decided to crop the image.
Our initial images looked as follows,

![full image](images/center_camera.png)

which was 320 by 160 pixels. We decided to crop the bottom and the top since
this does not hold any valuable information for driving. Our final image looks
as follows,

![cropped image](images/center_camera_crop.png)

this image is 320 by 90 pixels. To reduce the size of the network we rescale
the image to 80 by 80. We convert the image to YUV color space to reduce the
impact shadows on training.

### Data augmentation
We perform several data augmentation steps to increase the amount of training data.
First we use the center image, left and right image. To use the left and right image
we respectively add and subtract an uniformly randomly distributed number in
the range of 0.03 to 0.07. Since the circuit on which the car is trained tends
to contain more left turns there is less right turn training data. To
compensate for this we flip the image and take the negative angle. To increase
the robustness, and reduce overfitting, of network we rotate the non-flipped
center, right and left steering angle by factor between -5 and 5 degree. The
added benefit of flipping the data is that we have a more balanced data set. A key 
element when training neural networks.

In addition we randomly change the brightness of original center, right and left image by
converting to HSV color space and randomly multiplying the V channel.

### Training data generation
To generate the training data we drove four lapses where two were close to the
left yellow line and two close to the right yellow line with recovery. In
addition we drove drove three lapses with center line.

# Running the program

## Install requirements
You can install the requirements using `pip3 install -r requirements.txt`.

## Train the model
You can download sample data using `./download_sample_data.sh`. Once the data is downloaded the neural net can be trained using
`python3 model.py --model . --data data/ --epochs 10 `.

After completing the epochs you can use the model as follows, `python3 drive.py model.json`. You can run the simulator.

### Files
* `model.py` allows to train the weights of a neural network.
* `drive.py` allows to use the trained model to drive a car using the UDacity simulator.
* `model.h5` the weights of a model.
* `model.json` the json representation of the neural network.
* `requirements.txt` you can use this file to install all the python3 required files.
