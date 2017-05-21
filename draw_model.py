import argparse
import json

from keras.models import model_from_json
from keras.utils import plot_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model drawing')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        # model = model_from_json(json.loads(jfile.read()))
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    plot_model(model, to_file='model.png')
