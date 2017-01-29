import argparse
# import keras
import numpy as np
import pandas as pd
from skimage import io

def load_images(files, data_path):
    def load_func(files):
        for fl in files:
            img = io.imread('%s/%s' % (data_path, fl.strip()))
            yield img

    return np.stack(load_func(files), axis=0)


def build_model(model_path, data_path):
    # Load the log file.
    drive_log = pd.read_csv("%s/driving_log.csv" % (data_path))

    # Load the left, center and right images that come from the simulator,
    # these are the training features.
    print("Loading training images:")
    left_images = load_images(drive_log['left'], data_path)
    print("left [%s, %s, %s, %s]" % (left_images.shape))
    center_images = load_images(drive_log['center'], data_path)
    print("center [%s, %s, %s, %s]" % (center_images.shape))
    right_images = load_images(drive_log['right'], data_path)
    print("right [%s, %s, %s, %s]" % (right_images.shape))

    # Load the training labels.
    steering = drive_log['steering']
    throttle = drive_log['throttle']
    brake = drive_log['brake']
    speed = drive_log['speed']




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavior cloning training')
    parser.add_argument('--model', dest='model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--data', dest='data', type=str,
                        help='Path to data that should be used to train model')
    args = parser.parse_args()
    build_model(args.model, args.data)


