# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-05T09:07:49.359329Z","iopub.execute_input":"2021-08-05T09:07:49.359704Z","iopub.status.idle":"2021-08-05T09:07:53.061279Z","shell.execute_reply.started":"2021-08-05T09:07:49.359671Z","shell.execute_reply":"2021-08-05T09:07:53.060499Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-05T09:07:53.062561Z","iopub.execute_input":"2021-08-05T09:07:53.062976Z","iopub.status.idle":"2021-08-05T09:08:00.320983Z","shell.execute_reply.started":"2021-08-05T09:07:53.062946Z","shell.execute_reply":"2021-08-05T09:08:00.319617Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-05T09:08:00.323233Z","iopub.execute_input":"2021-08-05T09:08:00.323782Z","iopub.status.idle":"2021-08-05T09:08:05.193925Z","shell.execute_reply.started":"2021-08-05T09:08:00.323734Z","shell.execute_reply":"2021-08-05T09:08:05.192722Z"}}
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
    batch_size=48,
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
base_model.trainable = False ## trainable weights

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-05T09:08:05.197886Z","iopub.execute_input":"2021-08-05T09:08:05.198224Z","iopub.status.idle":"2021-08-05T09:08:05.351097Z","shell.execute_reply.started":"2021-08-05T09:08:05.198194Z","shell.execute_reply":"2021-08-05T09:08:05.350143Z"}}
from tensorflow.keras import layers, models

# adding our custom layers to this specific scope
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(50, activation='relu')
dense_layer_3 = layers.Dense(50, activation='relu')
dense_layer_4 = layers.Dense(40, activation='relu')
dense_layer_5 = layers.Dense(20, activation='relu')
dense_layer_6 = layers.Dense(20, activation='relu')
dense_layer_7 = layers.Dense(20, activation='relu')
dense_layer_8 = layers.Dense(20, activation='relu')
dense_layer_9 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(2, activation='softmax')

# A Sequential model is appropriate for a plain stack of layers where each
# layer has exactly one input tensor and one output tensor

# Creating a Sequential model
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_layer_3,
    dense_layer_4,
    dense_layer_5,
    dense_layer_6,
    dense_layer_7,
    dense_layer_8,
    dense_layer_9,
    prediction_layer
])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-05T09:08:05.352312Z","iopub.execute_input":"2021-08-05T09:08:05.352646Z"}}
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
history = model.fit(train_generator, epochs=20, callbacks=None,validation_data=val_generator)

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

# %% [code] {"jupyter":{"outputs_hidden":false}}
class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

model.save(
    './assets/n_model.h5',
    overwrite=True,
    include_optimizer=True,
    save_format="h5",
    signatures=None,
    options=None,
    save_traces=True,
)

print(bcolors.OK + "File Saved Successfully!" + bcolors.RESET)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return [idx,array[idx]]

def get_predicted_value(label,predicted):
    actual = "Normal" if find_nearest(label,1)[0] == 0 else "Pneumonia"
    prediction = find_nearest(predicted,1)
    prediction_class = "Normal" if prediction[0] == 0 else "Pneumonia"
    prediction_value = prediction[1] * 100
    return [actual,prediction_class,prediction_value]



for _ in range(1):
    img, label = val_generator.next()
    for i in range(10):
        plt.imshow(img[i])
        plt.show()
        [actual,prediction_class,prediction_value] = get_predicted_value(label[i],model.predict(img)[i])
        if(actual == prediction_class):
            color = bcolors.OK
        else:
            color = bcolors.FAIL
        print(color + f'predicted {prediction_class} with vaule {prediction_value}\nactual {actual}' + bcolors.RESET)



# %% [code] {"jupyter":{"outputs_hidden":false}}
print(label)

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_dataset_metadata_df = metadata_df[metadata_df.Dataset_type.isin(['TRAIN'])]
train_dataset_metadata_df = train_dataset_metadata_df.sort_values(by="X_ray_image_name")

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_dataset_metadata_df = metadata_df[metadata_df.Dataset_type.isin(['TEST'])]
test_dataset_metadata_df = test_dataset_metadata_df.sort_values(by="X_ray_image_name")

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_dataset_metadata_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_dataset_metadata_df.tail()

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_dataset_metadata_df.head(100)

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_dataset_metadata_df.tail(100)
