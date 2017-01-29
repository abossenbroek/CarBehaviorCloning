import argparse
import keras
import numpy as np
import pandas as pd
from skimage import io

def load_images(files, data_path):
    def load_func(files):
        for fl in files:
            img = io.imread("%s/%s" % (data_path, fl))
            yield img

    return np.stack(load_func(files), axis=0)



def build_model(model_path, data_path):
    drive_log = pd.read_csv("%s/driving_log.csv" % (model_path))

    center_images = load_images(drive_log['center'], data_path)
    left_images = load_images(drive_log['left'], data_path)
    right_images = load_images(drive_log['right'], data_path)

    print("Found the following training images:")
    print("left %s" % (left_images.shape))
    print("center %s" % (center_images.shape))
    print("right %s" % (right_images.shape))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavior cloning training')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('data', type=str,
                        help='Path to data that should be used to train model')
    args = parser.parse_args()
    build_model(args.model, args.data)


