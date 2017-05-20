import argparse
import numpy as np
import cv2
import os.path
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

from keras.models import Model
from keras.layers import Conv2D
from keras.layers.merge import add
from keras.layers import Dropout, Flatten, Dense, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers.pooling import AveragePooling2D
from keras.constraints import max_norm


MISSING = object()


def nvidia_model(input):
    x = Conv2D(32, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(input)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(32, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D(pool_size=(4, 4), padding='same')(x)
    resnet_in = x
    x = Conv2D(32, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    resnet_in = x
    x = Conv2D(32, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(64, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(256, (3, 3), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid')(x)

    x = Flatten()(x)
    #x = Dense(1164, activation="elu",
              #kernel_constraint=max_norm(2.),
              #kernel_initializer='glorot_normal')(x)
    #x = Dense(512,
    #          kernel_constraint=max_norm(2.),
    #          kernel_initializer='glorot_normal')(x)
    x = Dense(100,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dense(50,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    return x


def preprocess_img(file, fromColorSpace="BGR"):
    if not os.path.isfile(file):
        print("%s could not be found" % file)

    img = cv2.imread(file)
    if fromColorSpace == "BGR":
        convert = cv2.COLOR_BGR2YUV
    else:
        convert = cv2.COLOR_RGB2YUV
    img = cv2.cvtColor(img, convert)
    img = img[50:140, :, :]
    img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, np.float32)
    return img


# From https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_br = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_br
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(cv2.cvtColor(image1, cv2.COLOR_HSV2RGB),
                          cv2.COLOR_RGB2YUV)
    return image1

def rotate_image(image, max_angle = 5):
    rows, cols, colors = image.shape
    rot_ang = np.random.uniform(-max_angle, max_angle)
    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), rot_ang, 1)
    rot_image = cv2.warpAffine(image, rot_mat, (cols, rows))

    return rot_image


def process_line(line, log_path):
    c_image_nm, l_image_nm, r_image_nm, ang, throttle, brake, speed = line.split(",")

    # Load the images
    c_img = preprocess_img(log_path + "/" + c_image_nm.strip())

    # Generate the angs
    c_ang = float(ang)
    c_flp_ang = -c_ang

    # Always add a flipped image
    c_flp_img = np.fliplr(c_img)

    x = np.concatenate((c_img[np.newaxis, ...],
                        c_flp_img[np.newaxis, ...],), axis=0)

    y = np.vstack((c_ang, c_flp_ang,))

    return [x, y]


def generate_input_from_file(log_file, log_path):
    while 1:
        f = open(log_file)
        for line in f:
            x, y = process_line(line, log_path)
            yield (x, y)
        f.close()


def load_original_file(log_file, log_path):
    X = []
    Y = []

    f = open(log_file)
    for i, line in enumerate(tqdm(f)):
        if i < 1:
            continue
        else:
            x, y = process_line(line, log_path)
            X.append(x)
            Y.append(y)
    f.close()

    return [np.concatenate(X, axis=0), np.concatenate(Y, axis=0)]


def balance_data_set(X, y):
    # We seek to balance the data set such that y has equal groupings. For that
    # we will take the following steps:
    # 1. discretize y
    # 2. remove the mode of y since this is probably very frequent data
    #    driving straight.
    # 3. add an id to the data set
    # 4.

    # Determine the bins that we will use.
    y_lins = np.linspace(min(y), max(y), 200)
    # Bin the angle.
    y_dig = np.digitize(y, y_lins)
    non_mode_idx = y_dig != stats.mode(y_dig)[0]
    # Determine the index of X
    X_idx = np.arange(0, X.shape[0]).reshape(-1, 1)
    # Let us drop all observations that have a mode value
    X_idx = X_idx[non_mode_idx]
    y_dig = y_dig[non_mode_idx]

    # Perform random over sampling.
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_sample(X_idx, y_dig)

    # We now have all our indices balanced. We obviously can have a lot of double entries which may cause the neural net
    # to learn certain aspects of an image.

    # Let us first create the data set.
    X_b = X[y_res[0]]
    y_b = y[y_res[0]]

    return list(X_b, y_b)





def build_model(model_path, data_path, epochs, new_data=MISSING,
                model_file=MISSING):
    # Load the log file.
    drive_log = "%s/driving_log.csv" % data_path

    X, y = load_original_file(drive_log, data_path)

    X, y = balance_data_set(X, y)

    #X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])

    img_input = Input(shape=X.shape[1:], dtype='float32', name="images")

    input = Lambda(lambda x: x/127.5 - 1.0)(img_input)

    x = nvidia_model(input)

    steering_output = Dense(1, activation='linear', name='steering_output')(x)

    model = Model(inputs=img_input,
                  outputs=steering_output)

    if model_file is not MISSING:
        print("Loading model from %s" % model_file)
        with open(model_file, 'r') as jfile:
            model = model_from_json(jfile.read())

        weights_file = model_file.replace('json', 'h5')
        model.load_weights(weights_file)

    print(model.summary())

    model.compile(optimizer='nadam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1,
                                   save_best_only=True)
    csv_logger = CSVLogger('training.log')

    model.fit(x=X,
              y=y,
              epochs=epochs, batch_size=1024, validation_split=0.20,
              shuffle=True,
              callbacks=[early_stopping,
                         checkpointer,
                         csv_logger])

    model_json_file = "%s/model.json" % (model_path)
    model_weights_file = "%s/model.h5" % (model_path)
    print("About to save model to '%s'" % (model_json_file))
    print("About to save model weights to '%s'" % (model_weights_file))

    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)

    print("Successfully saved JSON file")
    model.save_weights(model_weights_file)
    print("Successfully saved weights")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavior cloning training')
    parser.add_argument('--model', dest='model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--data', dest='data', type=str,
                        help='Path to data that should be used to train model')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='Path to data that should be used to train model')
    parser.add_argument('--load', dest='load', type=str,
                        help='Name of the model to load to perform transfer learning.')

    args = parser.parse_args()
    if args.load:
        build_model(args.model, args.data, args.epochs, args.load)
    else:
        build_model(args.model, args.data, args.epochs, MISSING)

