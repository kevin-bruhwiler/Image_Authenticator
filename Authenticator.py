from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalMaxPooling2D
from keras.optimizers import rmsprop
from keras.callbacks import ModelCheckpoint, Callback
import os, random
import cv2
from skimage import restoration
from scipy.ndimage import imread
import numpy as np

_AUTHENTIC_IMAGES = "Au"
_TAMPERED_IMAGES = "Tp"
_VAL_AUTHENTIC_IMAGES = "Val_Au"
_VAL_TAMPERED_IMAGES = "Val_Tp"
_BATCH_SIZE = 1
_CHANNELS = 15
_MODEL = "authenticator_weights"+str(_CHANNELS)+".hdf5"


class LossHistory(Callback):
    def __init__(self, file):
        super(LossHistory, self).__init__()
        self.file = file

    def on_epoch_end(self, epoch, logs={}):
        with open(self.file, "a") as file:
            file.write(str("%.5f"%logs.get("loss") + "   :   " + "%.5f"%logs.get("acc")
                           + "%.5f"%logs.get("val_loss") + "   :   " + "%.5f"%logs.get("val_acc") + os.linesep))


def shuffle(x, y):
    data = list(zip(x,y))
    random.shuffle(data)
    return zip(*data)


def validation_data_generator():
    inp = os.listdir(_VAL_AUTHENTIC_IMAGES)
    out = [0] * len(inp)
    inp.extend(os.listdir(_VAL_TAMPERED_IMAGES))
    out.extend([1] * (len(inp) - len(out)))

    while True:
        inp, out = shuffle(inp, out)
        for i in range(0, len(inp), _BATCH_SIZE):
            data = inp[i:i + _BATCH_SIZE]
            x = []
            for d in data:
                try:
                    x.append(process_image(imread(_VAL_AUTHENTIC_IMAGES + "/" + d)))
                    if len(x[-1].shape) is not 3 or x[-1].shape[2] is not _CHANNELS:
                        x = x[:-1]
                except FileNotFoundError:
                    x.append(process_image(imread(_VAL_TAMPERED_IMAGES + "/" + d)))
                    if len(x[-1].shape) is not 3 or x[-1].shape[2] is not _CHANNELS:
                        x = x[:-1]
            if len(x) is _BATCH_SIZE:
                yield np.asarray(x), np.asarray(out[i:i+_BATCH_SIZE])


def process_image(img):
    shp = img.shape
    for i in range(shp[-1]):
        blr = cv2.blur(img[...,i], (3, 3))
        img = np.concatenate((img, (img[...,i]-blr).reshape(shp[0], shp[1], 1)), axis=-1)

        blr = cv2.GaussianBlur(img[...,i].reshape(shp[0], shp[1], 1),(3,3),0.5)
        img = np.concatenate((img, (img[...,i]-blr).reshape(shp[0], shp[1], 1)), axis=-1)

        blr = cv2.medianBlur(img[...,i].reshape(shp[0], shp[1], 1).astype(np.float32), 3)
        img = np.concatenate((img, (img[...,i]-blr).reshape(shp[0], shp[1], 1)), axis=-1)

        psf = np.ones((3, 3)) / 9
        denoise = restoration.unsupervised_wiener(img[...,i], psf)[0]
        img = np.concatenate((img, (img[...,i]-denoise).reshape(shp[0], shp[1], 1)), axis=-1)

        #psf = np.ones((5, 5)) / 25
        #denoise = restoration.unsupervised_wiener(img[..., i], psf)[0]
        #img = np.concatenate((img, (img[..., i]-denoise).reshape(shp[0], shp[1], 1)), axis=-1)
    return img


def data_generator():
    inp = os.listdir(_AUTHENTIC_IMAGES)
    out = [0] * len(inp)
    inp.extend(os.listdir(_TAMPERED_IMAGES))
    out.extend([1] * (len(inp) - len(out)))

    while True:
        inp, out = shuffle(inp, out)
        for i in range(0, len(inp), _BATCH_SIZE):
            data = inp[i:i+_BATCH_SIZE]
            x = []
            for d in data:
                try:
                    x.append(process_image(imread(_AUTHENTIC_IMAGES+"/"+d)))
                    if len(x[-1].shape) is not 3 or x[-1].shape[2] is not _CHANNELS:
                        x = x[:-1]
                except FileNotFoundError:
                    x.append(process_image(imread(_TAMPERED_IMAGES + "/" + d)))
                    if len(x[-1].shape) is not 3 or x[-1].shape[2] is not _CHANNELS:
                        x = x[:-1]
            if len(x) is _BATCH_SIZE:
                yield np.asarray(x), np.asarray(out[i:i+_BATCH_SIZE])


def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(None, None, _CHANNELS)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=rmsprop(.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def preprocess():
    authentic_images = os.listdir(_AUTHENTIC_IMAGES)
    tampered_images = os.listdir(_TAMPERED_IMAGES)
    val_authentic_images = os.listdir(_VAL_AUTHENTIC_IMAGES)
    val_tampered_images = os.listdir(_VAL_TAMPERED_IMAGES)
    return len(authentic_images) + len(tampered_images), len(val_authentic_images) + len(val_tampered_images)


def run():
    num, num_val = preprocess()
    model = build_model()
    checkpoint = ModelCheckpoint(_MODEL, verbose=1, monitor='val_loss', save_best_only=True)
    log = LossHistory("log"+str(_CHANNELS)+".txt")
    callback_list = [checkpoint, log]
    if os.path.isfile(_MODEL):
        model.load_weights(_MODEL)
    try:
        model.fit_generator(data_generator(), int(num/_BATCH_SIZE), epochs=1000, callbacks=callback_list, max_queue_size=50,
                            use_multiprocessing=True, validation_data=validation_data_generator(), validation_steps=int(num_val/_BATCH_SIZE))
    except KeyboardInterrupt:
        model.save(_MODEL)


if __name__=='__main__':
    run()
