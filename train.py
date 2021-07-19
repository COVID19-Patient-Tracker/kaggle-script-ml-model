# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T17:52:55.292714Z","iopub.execute_input":"2021-07-18T17:52:55.293076Z","iopub.status.idle":"2021-07-18T17:52:55.328604Z","shell.execute_reply.started":"2021-07-18T17:52:55.293043Z","shell.execute_reply":"2021-07-18T17:52:55.327698Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
#         print()

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T17:52:55.330372Z","iopub.execute_input":"2021-07-18T17:52:55.330731Z","iopub.status.idle":"2021-07-18T17:52:55.336986Z","shell.execute_reply.started":"2021-07-18T17:52:55.330692Z","shell.execute_reply":"2021-07-18T17:52:55.335791Z"}}
# import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T17:52:55.370963Z","iopub.execute_input":"2021-07-18T17:52:55.371336Z","iopub.status.idle":"2021-07-18T17:52:57.787408Z","shell.execute_reply.started":"2021-07-18T17:52:55.371305Z","shell.execute_reply":"2021-07-18T17:52:57.786513Z"}}
# train dataset file path
train_file_path = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'

# path to csv file, contains labels ['Normal','pnemonia']
metadata_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

# Generate batches of tensor image data with real-time data augmentation for train.
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generate batches of tensor image data with real-time data augmentation for validation.
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

# Takes the dataframe and the path to a directory + generates batches.
# The generated batches contain augmented/normalized data.
train_generator = train_datagen.flow_from_dataframe(
    dataframe = metadata_df,
    directory=train_file_path,
    x_col="X_ray_image_name",
    y_col="Label",
    weight_col=None,
    target_size=(224, 224),
    color_mode="rgb",
    classes=["Normal","Pnemonia"],
    class_mode="categorical",
    batch_size=20,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="jpeg",
    subset='training',
    interpolation="nearest",
    validate_filenames=True
)

# Takes the dataframe and the path to a directory + generates batches.
# The generated batches contain augmented/normalized data.
val_generator = val_datagen.flow_from_dataframe(
    dataframe = metadata_df,
    directory=train_file_path,
    x_col="X_ray_image_name",
    y_col="Label",
    weight_col=None,
    target_size=(224, 224),
    color_mode="rgb",
    classes=["Normal","Pnemonia"],
    class_mode="categorical",
    batch_size=10,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix="",
    save_format="jpeg",
    subset='validation',
    interpolation="nearest",
    validate_filenames=True
)

# display samples . display data on one batch
# lable => [1,0] = 'Normal' or [0,1] = 'Pnemonia'
# for _ in range(1):
#     img, label = val_generator.next()
#     print(img.shape)   #  (32,224,224,3)
#     for i in range(10):
#         plt.imshow(img[i])
#         plt.show()
#         print(label[i])

img, label = train_generator.next()
# val_img,val_label = val_generator.next()
# print(label)

## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=img[0].shape)
base_model.trainable = True ## trainable weights

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T17:52:57.789015Z","iopub.execute_input":"2021-07-18T17:52:57.789368Z","iopub.status.idle":"2021-07-18T17:52:57.880272Z","shell.execute_reply.started":"2021-07-18T17:52:57.789329Z","shell.execute_reply":"2021-07-18T17:52:57.879293Z"}}
from tensorflow.keras import layers, models

# adding our custom layers to this specific scope
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(40, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(2, activation='softmax')

# A Sequential model is appropriate for a plain stack of layers where each
# layer has exactly one input tensor and one output tensor

# Creating a Sequential model
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T17:52:57.88222Z","iopub.execute_input":"2021-07-18T17:52:57.882662Z","iopub.status.idle":"2021-07-18T18:06:00.230676Z","shell.execute_reply.started":"2021-07-18T17:52:57.882616Z","shell.execute_reply":"2021-07-18T18:06:00.229903Z"}}
from tensorflow.keras.callbacks import EarlyStopping

# compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# use earlystopping strategy to prevent overfitting
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)


# train_ds,train_labels = train_generator.next() 

# train the model
history = model.fit(train_generator, epochs=10, callbacks=None,validation_data=val_generator)

# historyA =model.evaluate(val_generator)

# plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:00.232289Z","iopub.execute_input":"2021-07-18T18:06:00.232632Z","iopub.status.idle":"2021-07-18T18:06:01.113942Z","shell.execute_reply.started":"2021-07-18T18:06:00.232597Z","shell.execute_reply":"2021-07-18T18:06:01.112881Z"}}
model.save(
    './assets/n_model.h5',
    overwrite=True,
    include_optimizer=True,
    save_format="h5",
    signatures=None,
    options=None,
    save_traces=True,
)
img, label = train_generator.next()

model.predict(img)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.115795Z","iopub.execute_input":"2021-07-18T18:06:01.116268Z","iopub.status.idle":"2021-07-18T18:06:01.123505Z","shell.execute_reply.started":"2021-07-18T18:06:01.116221Z","shell.execute_reply":"2021-07-18T18:06:01.122205Z"}}
print(label)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.12528Z","iopub.execute_input":"2021-07-18T18:06:01.125787Z","iopub.status.idle":"2021-07-18T18:06:01.145023Z","shell.execute_reply.started":"2021-07-18T18:06:01.125741Z","shell.execute_reply":"2021-07-18T18:06:01.144116Z"}}
train_dataset_metadata_df = metadata_df[metadata_df.Dataset_type.isin(['TRAIN'])]
train_dataset_metadata_df = train_dataset_metadata_df.sort_values(by="X_ray_image_name")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.14644Z","iopub.execute_input":"2021-07-18T18:06:01.146801Z","iopub.status.idle":"2021-07-18T18:06:01.156055Z","shell.execute_reply.started":"2021-07-18T18:06:01.146764Z","shell.execute_reply":"2021-07-18T18:06:01.155171Z"}}
test_dataset_metadata_df = metadata_df[metadata_df.Dataset_type.isin(['TEST'])]
test_dataset_metadata_df = test_dataset_metadata_df.sort_values(by="X_ray_image_name")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.158865Z","iopub.execute_input":"2021-07-18T18:06:01.159253Z","iopub.status.idle":"2021-07-18T18:06:01.177583Z","shell.execute_reply.started":"2021-07-18T18:06:01.159214Z","shell.execute_reply":"2021-07-18T18:06:01.176473Z"}}
train_dataset_metadata_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.179581Z","iopub.execute_input":"2021-07-18T18:06:01.180043Z","iopub.status.idle":"2021-07-18T18:06:01.19861Z","shell.execute_reply.started":"2021-07-18T18:06:01.180002Z","shell.execute_reply":"2021-07-18T18:06:01.197567Z"}}
train_dataset_metadata_df.tail()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.199971Z","iopub.execute_input":"2021-07-18T18:06:01.200339Z","iopub.status.idle":"2021-07-18T18:06:01.222566Z","shell.execute_reply.started":"2021-07-18T18:06:01.200291Z","shell.execute_reply":"2021-07-18T18:06:01.221612Z"}}
test_dataset_metadata_df.head(100)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-18T18:06:01.223803Z","iopub.execute_input":"2021-07-18T18:06:01.224132Z","iopub.status.idle":"2021-07-18T18:06:01.245433Z","shell.execute_reply.started":"2021-07-18T18:06:01.224102Z","shell.execute_reply":"2021-07-18T18:06:01.244396Z"}}
test_dataset_metadata_df.tail(100)
