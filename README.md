# CarBehaviorCloning
Behavior cloning of driving a car on a track.

## Install requirements
You can install the requirements using `pip3 install -r requirements.txt`.

## Train the model
You can download sample data using `./download_sample_data.sh`. Once the data is downloaded the neural net can be trained using 
`python3 model.py --model . --data data/ --epochs 10`.

After completing the epochs you can use the model as follows, `python3 drive.py model.json`. You can run the simulator.

## Current issue
Car keeps crashing in walls.

### Files
* `model.py` allows to train the weights of a neural network.
* `drive.py` allows to use the trained model to drive a car using the UDacity simulator.
* `model.h5` the weights of a model.
* `model.json` the json representation of the neural network.
* `requirements.txt` you can use this file to install all the python3 required files.

### Background literature
[NVidia's article on a pipeline](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
