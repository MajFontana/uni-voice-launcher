import tensorflow as tf
from tensorflow.keras import layers, models, utils
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import keras_tuner as kt



RAND_SEED = 69
TRAIN_SPLIT = 0.5
SAMP_RATE = 16000
REC_DURATION = 2
DATA_DIR = "samples/"
CLASSES = ["python", "notepad", "terminal", "documents"]
EPOCHS = 50
MODEL_FILE = "model"
REPEAT_COUNT = 20
BATCH_SIZE = 32
SHIFT_AMOUNT = 3200
MIN_AMPLITUDE_SCALE = 0.25
MAX_NOISE = 0.2
WINDOW_SIZE = 255
WINDOW_STEP = 128



# load data
seq_size = int(SAMP_RATE * REC_DURATION)

dataset = tf.keras.utils.audio_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    batch_size=BATCH_SIZE,
    output_sequence_length=seq_size,
    seed=RAND_SEED
)



# preprocess - reshape
def flatten_channel_dim(data):
    data = tf.squeeze(data, axis=-1)
    return data

dataset = dataset.map(lambda d, l: (flatten_channel_dim(d), l), tf.data.AUTOTUNE)



# preprocess - increase sample count and add variation
batches = tfds.as_numpy(dataset)
data_arr = np.zeros([0, seq_size])
labels_arr = np.array([])
for batch in batches:
    data, labels = batch
    data_arr = np.concatenate([data_arr, data], axis=0)
    labels_arr = np.concatenate([labels_arr, labels])

remainder = BATCH_SIZE - (REPEAT_COUNT * len(labels_arr) % BATCH_SIZE)
data_patch = data_arr[:remainder]
labels_patch = labels_arr[:remainder]
data_arr = np.repeat(data_arr, REPEAT_COUNT, axis=0)
labels_arr = np.repeat(labels_arr, REPEAT_COUNT)
data_arr = np.concatenate([data_arr, data_patch])
labels_arr = np.concatenate([labels_arr, labels_patch])

dataset = tf.data.Dataset.from_tensor_slices((data_arr, labels_arr))
dataset = dataset.batch(BATCH_SIZE)

np.random.seed(RAND_SEED)

def variation(data):
    # shift in time
    sh = np.random.randint(-SHIFT_AMOUNT, SHIFT_AMOUNT)
    data = np.concatenate([data[:, sh:], data[:, :sh]], axis=1)

    # different amplitudes
    amp = np.random.rand(data.shape[0]) * (1 - MIN_AMPLITUDE_SCALE) + MIN_AMPLITUDE_SCALE
    data = data * amp[:, None]

    # noise
    namp = np.random.rand(data.shape[0]) * MAX_NOISE
    noise = np.random.rand(*data.shape) * namp[:, None]
    data = data + noise

    data = data.astype(np.float32)
    return data

dataset = dataset.map(lambda d, l: (tf.numpy_function(variation, [d], tf.float32), l), tf.data.AUTOTUNE)



# preprocess - spectogram
def stft(data, labels):
    spec = tf.signal.stft(data, frame_length=WINDOW_SIZE, frame_step=WINDOW_STEP)
    spec = tf.abs(spec)
    spec = spec[..., tf.newaxis] # image channel dimension
    return spec, labels

dataset = dataset.map(stft, tf.data.AUTOTUNE)



# split dataset & loading optimization
train_set, val_set = utils.split_dataset(dataset, TRAIN_SPLIT)

shuffle_size = len(train_set)
train_set = train_set.cache().shuffle(RAND_SEED).prefetch(tf.data.AUTOTUNE)
val_set = val_set.cache().prefetch(tf.data.AUTOTUNE)



# create model
for ex, _ in train_set.take(1):
    break
in_shape = ex.shape[1:]

norm_layer = layers.Normalization()
norm_layer.adapt(data=train_set.map(map_func=lambda d, l: d))


def builder(hyparam):

    resize_x = hyparam.Int("resize_x", min_value=16, max_value=128, step=16)
    resize_y = hyparam.Int("resize_y", min_value=16, max_value=128, step=16)
    conv1_size = hyparam.Int("conv1_size", min_value=16, max_value=128, step=16)
    conv1_kern = hyparam.Int("conv1_kern", min_value=3, max_value=7, step=2)
    conv1_act = hyparam.Choice("conv1_act", ["relu", "selu", "sigmoid"])
    conv2_size = hyparam.Int("conv2_size", min_value=0, max_value=128, step=16)
    conv2_kern = hyparam.Int("conv2_kern", min_value=3, max_value=7, step=2)
    conv2_act = hyparam.Choice("conv2_act", ["relu", "selu", "sigmoid"])
    conv3_size = hyparam.Int("conv3_size", min_value=0, max_value=128, step=16)
    conv3_kern = hyparam.Int("conv3_kern", min_value=3, max_value=7, step=2)
    conv3_act = hyparam.Choice("conv3_act", ["relu", "selu", "sigmoid"])
    dense1_size = hyparam.Int("dense1_size", min_value=0, max_value=128, step=16)
    dense1_act = hyparam.Choice("dense1_act", ["relu", "selu", "sigmoid"])
    dense2_size = hyparam.Int("dense2_size", min_value=0, max_value=128, step=16)
    dense2_act = hyparam.Choice("dense2_act", ["relu", "selu", "sigmoid"])
    pool_x = hyparam.Int("pool_x", min_value=2, max_value=5, step=1)
    pool_y = hyparam.Int("pool_y", min_value=2, max_value=5, step=1)
    drop1 = hyparam.Float("drop1", min_value=0, max_value=0.8, step=0.1)
    drop2 = hyparam.Float("drop2", min_value=0, max_value=0.8, step=0.1)
    drop3 = hyparam.Float("drop3", min_value=0, max_value=0.8, step=0.1)

    model = models.Sequential([
        layers.Input(shape=in_shape),
        layers.Resizing(resize_x, resize_y),
        norm_layer,
        layers.Conv2D(conv1_size, conv1_kern, activation=conv1_act),
        layers.Conv2D(conv2_size, conv2_kern, activation=conv2_act),
        layers.Conv2D(conv3_size, conv3_kern, activation=conv3_act),
        layers.MaxPooling2D([pool_x, pool_y]),
        layers.Dropout(drop1),
        layers.Flatten(),
        layers.Dense(dense1_size, activation=dense1_act),
        layers.Dropout(drop2),
        layers.Dense(dense2_size, activation=dense2_act),
        layers.Dropout(drop3),
        layers.Dense(len(CLASSES)),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model



# Get optimal hyperparameters
tuner = kt.Hyperband(builder,
                     objective="val_accuracy",
                     max_epochs=20,
                     factor=3,
                     directory="tuner",
                     project_name="tuning_test")

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

tuner.search(x=train_set, validation_data=val_set, epochs=50, callbacks=[stop_early])
best_config = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)



# train model
history = model.fit(
    x=train_set,
    validation_data=val_set,
    epochs=50
#    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
)

accu = history.history["val_accuracy"]
best_epoch = accu.index(max(accu)) + 1

model = tuner.hypermodel.build(best_config)
history = model.fit(
    x=train_set,
    validation_data=val_set,
    epochs=best_epoch
#    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
)

model.save(MODEL_FILE)
plt.show()
