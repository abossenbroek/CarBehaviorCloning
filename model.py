import argparse
import numpy as np
import cv2
import os.path
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Conv2D
from keras.layers.merge import add
from keras.layers import Dropout, Flatten, Dense, Input, Lambda, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers.pooling import AveragePooling2D
from keras.constraints import max_norm
from keras.preprocessing.image import ImageDataGenerator

MISSING = object()


def complex_model(input):
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
    x = Dense(1164, activation="elu",
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dense(512,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dense(100,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dense(50,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    return x

def nvidia_model(input):
    x = Conv2D(24, (5, 5), padding='same', strides= (2, 2),
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal',
               activation='elu')(input)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(36, (5, 5), padding='same', strides= (2, 2),
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal',
               activation='elu')(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(48, (5, 5), padding='same', strides= (2, 2),
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal',
               activation='elu')(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal',
               activation='elu')(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (5, 5), padding='same',
               kernel_constraint=max_norm(2.),
               kernel_initializer='glorot_normal',
               activation='elu')(x)
    x = SpatialDropout2D(0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation="elu",
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dense(50,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dense(10,
              kernel_constraint=max_norm(2.),
              kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    return x


def preprocess_img(file, fromColorSpace="BGR"):
    if not os.path.isfile(file):
        print("%s could not be found" % file)

    img = cv2.imread(file)
    if fromColorSpace == "BGR":
        convert = cv2.COLOR_BGR2RGB
    img = cv2.cvtColor(img, convert)
    img = img[55:140, :, :]
    img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img, np.float32)
    return img


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


def balance_data_set(X, y, bins = 200):
    # Determine the bins that we will use.
    hist, y_bins = np.histogram(y, bins = bins)
    # Bin the angle.
    y_dig = np.digitize(y, y_bins)
    non_mode_idx = np.where(y_dig != stats.mode(y_dig)[0])
    # Determine the index of X
    X_idx = np.arange(0, X.shape[0]).reshape(-1, 1)


    y_dig = y_dig[non_mode_idx].reshape(-1, 1)
    X_idx = X_idx[non_mode_idx].reshape(-1, 1)

    # Perform random over sampling.
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_sample(X_idx, y_dig.ravel())

   # Let us first create the data set.
    X_b = X[X_res]
    y_b = y[X_res]
    # Reshape to proper size.
    X_b = X_b.reshape(X_b.shape[0], X_b.shape[2], X_b.shape[3], X_b.shape[4])
    y_b = y_b.reshape(y_b.shape[0])

    assert(X_b.shape[0] == y_b.shape[0])

    return [X_b, y_b]


def build_model(model_path, data_path, epochs, new_data=MISSING,
                model_file=MISSING):
    # Load the log file.
    drive_log = "%s/driving_log.csv" % data_path

    X, y = load_original_file(drive_log, data_path)
    print("Loaded %s data samples" % X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train sample has size %s" % X_train.shape[0])

    # Balance the data set.
    #X, y = balance_data_set(X_train, y_train)
    print("After balancing and removing mode we have %s data samples" % X.shape[0])

    img_gen = ImageDataGenerator(rotation_range=5,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=0.1*np.pi,
                                 zoom_range=[0.90,1.10],
                                 fill_mode='constant',
                                 channel_shift_range=0.9)
    img_gen.fit(X, seed=42)

    X_aug = []
    y_aug = []

    for X_batch, y_batch in img_gen.flow(X, y, batch_size=X.shape[0]):
        X_aug = X_batch
        y_aug = y_batch
        break

    print("X_aug shape", X_aug.shape)
    print("y_aug shape", y_aug.shape)

    X = X_aug
    y = y_aug

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1,
                                   save_best_only=True)
    csv_logger = CSVLogger('training.log')

    model.fit(x=X,
              y=y,
              epochs=epochs, batch_size=512, validation_data=[X_test, y_test],
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

