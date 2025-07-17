<a href="https://colab.research.google.com/github/SheryllD/diabetic_retinopathy_ml/blob/main/DR_ML.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Diabetic Retinnopathy | Machine Learning Project

### Libraries


```python
import numpy as np
import pandas as pd
import random, os
import shutil
import kagglehub
import os
import shutil
import cv2
import random
import pickle

import seaborn as sns

import tensorflow as tf
from tensorflow import lite
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy

import matplotlib.pyplot as plt
from matplotlib.image import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')
```

### Connecting with Kaggle


```python
# Install Kaggle API
!pip install -q Kaggle
!apt-get update
!apt-get install tree -y
```

    0% [Working]            Get:1 https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/ InRelease [3,632 B]
    0% [Connecting to archive.ubuntu.com (185.125.190.83)] [Waiting for headers] [1                                                                               Hit:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
    Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]
    Hit:4 http://archive.ubuntu.com/ubuntu jammy InRelease
    Get:5 https://r2u.stat.illinois.edu/ubuntu jammy InRelease [6,555 B]
    Get:6 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
    Get:7 https://r2u.stat.illinois.edu/ubuntu jammy/main all Packages [9,086 kB]
    Get:8 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1,264 kB]
    Hit:9 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
    Hit:10 https://ppa.launchpadcontent.net/graphics-drivers/ppa/ubuntu jammy InRelease
    Hit:11 https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu jammy InRelease
    Get:12 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [4,780 kB]
    Get:13 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]
    Get:14 https://r2u.stat.illinois.edu/ubuntu jammy/main amd64 Packages [2,755 kB]
    Get:15 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [3,098 kB]
    Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [3,413 kB]
    Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1,567 kB]
    Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [4,948 kB]
    Get:19 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [35.2 kB]
    Fetched 31.3 MB in 4s (7,947 kB/s)
    Reading package lists... Done
    W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    The following NEW packages will be installed:
      tree
    0 upgraded, 1 newly installed, 0 to remove and 38 not upgraded.
    Need to get 47.9 kB of archives.
    After this operation, 116 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu jammy/universe amd64 tree amd64 2.0.2-1 [47.9 kB]
    Fetched 47.9 kB in 0s (161 kB/s)
    Selecting previously unselected package tree.
    (Reading database ... 126281 files and directories currently installed.)
    Preparing to unpack .../tree_2.0.2-1_amd64.deb ...
    Unpacking tree (2.0.2-1) ...
    Setting up tree (2.0.2-1) ...
    Processing triggers for man-db (2.10.2-1) ...


### Setting up Kaggle & Downloading the Data


```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Setting up Kaggle credentials
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/Keys/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and unzip
!kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-gaussian-filtered --unzip
```

    Mounted at /content/drive
    Dataset URL: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered
    License(s): CC0-1.0
    Downloading diabetic-retinopathy-224x224-gaussian-filtered.zip to /content
     87% 372M/427M [00:04<00:00, 62.9MB/s]
    100% 427M/427M [00:04<00:00, 106MB/s] 



```python
# List directory after unzipping
print("Top-level files and folders:")
print(os.listdir())
```

    Top-level files and folders:
    ['.config', 'train.csv', 'gaussian_filtered_images', 'drive', 'sample_data']


### Loading the Data


```python
# Loading the CSV file into a pandas DataFrame
path = "/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered"
df = pd.read_csv("/content/train.csv")

diagnosis_dict_binary = {
    0: 'No_DR',
    1: 'DR',
    2: 'DR',
    3: 'DR',
    4: 'DR'
}

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

df['binary_type'] =  df['diagnosis'].map(diagnosis_dict_binary.get)
df['type'] = df['diagnosis'].map(diagnosis_dict.get)
print(df.head())

# Image folder root
image_root = os.path.join(path, 'gaussian_filtered_images', 'gaussian_filtered_images')

# Output base folder
output_root = "/content/processed"
```

            id_code  diagnosis binary_type            type
    0  000c1434d8d7          2          DR        Moderate
    1  001639a390f0          4          DR  Proliferate_DR
    2  0024cdab0c1e          1          DR            Mild
    3  002c21358ce6          0       No_DR           No_DR
    4  005b95c28852          0       No_DR           No_DR


### Creating Paths


```python
# Reviewing the paths
path = "/content" # Corrected path
print("Downloaded path:", path)
print("Contents of the folder:")
print(os.listdir(path))
```

    Downloaded path: /content
    Contents of the folder:
    ['.config', 'train.csv', 'gaussian_filtered_images', 'drive', 'sample_data']



```python
# Images
images = os.path.join(path, 'gaussian_filtered_images', 'gaussian_filtered_images')
print("Contents of the folder:", os.listdir(images))
```

    Contents of the folder: ['Moderate', 'export.pkl', 'Proliferate_DR', 'No_DR', 'Mild', 'Severe']



```python
# Loading the CSV
print(df.head(10))
```

            id_code  diagnosis binary_type            type
    0  000c1434d8d7          2          DR        Moderate
    1  001639a390f0          4          DR  Proliferate_DR
    2  0024cdab0c1e          1          DR            Mild
    3  002c21358ce6          0       No_DR           No_DR
    4  005b95c28852          0       No_DR           No_DR
    5  0083ee8054ee          4          DR  Proliferate_DR
    6  0097f532ac9f          0       No_DR           No_DR
    7  00a8624548a9          2          DR        Moderate
    8  00b74780d31d          2          DR        Moderate
    9  00cb6555d108          1          DR            Mild


### Viewing Data Distrubution


```python
df['type'].value_counts().plot(kind='bar')
```




    <Axes: xlabel='type'>




    
![png](DR_ML_files/DR_ML_16_1.png)
    



```python
df['binary_type'].value_counts().plot(kind='bar')
```




    <Axes: xlabel='binary_type'>




    
![png](DR_ML_files/DR_ML_17_1.png)
    


### Split Data to Train, Test & Validation test


```python
# Split into stratified train, val, and test sets
train_mid, val = train_test_split(df, test_size = 0.15, stratify = df['type'])
train, test = train_test_split(train_mid, test_size = 0.15 / (1 - 0.15), stratify = train_mid['type'])

print("Train Set:")
print(train['type'].value_counts(), '\n')

print("Test Set:")
print(test['type'].value_counts(), '\n')

print("Val Set:")
print(val['type'].value_counts(), '\n')
```

    Train Set:
    type
    No_DR             1263
    Moderate           699
    Mild               258
    Proliferate_DR     207
    Severe             135
    Name: count, dtype: int64 
    
    Test Set:
    type
    No_DR             271
    Moderate          150
    Mild               56
    Proliferate_DR     44
    Severe             29
    Name: count, dtype: int64 
    
    Val Set:
    type
    No_DR             271
    Moderate          150
    Mild               56
    Proliferate_DR     44
    Severe             29
    Name: count, dtype: int64 
    


### Creating Working Directories for Train, Validation and Test


```python
# Create working directories for train/val/test
base_dir = ''

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)
```


```python
src_dir = "/content/gaussian_filtered_images/gaussian_filtered_images"
print(os.listdir(src_dir))
```

    ['Moderate', 'export.pkl', 'Proliferate_DR', 'No_DR', 'Mild', 'Severe']


### Setting up the Directory


```python
# Setting up the source directory
src_dir = "/content/gaussian_filtered_images/gaussian_filtered_images"  # actual path in Colab

# Setting up destination working directories
base_dir = "/content/processed"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Recreateing destination folders
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

os.makedirs(train_dir)
os.makedirs(val_dir)
os.makedirs(test_dir)

# Defining a function to copy the data
def copy_split(df, split_dir):
    for _, row in df.iterrows():
        diagnosis = row['type']
        binary = row['binary_type']
        filename = row['id_code'] + ".png"

        srcfile = os.path.join(src_dir, diagnosis, filename)
        dst_folder = os.path.join(split_dir, binary)

        os.makedirs(dst_folder, exist_ok=True)

        if os.path.exists(srcfile):
            shutil.copy(srcfile, dst_folder)
        else:
            print(f"[Missing] {srcfile}")

# Copying all splits
copy_split(train, train_dir)
copy_split(val, val_dir)
copy_split(test, test_dir)
```

#### Viewing a Sample of the Train Set


```python
sample_folder = "/content/processed/train/DR"
sample_images = os.listdir(sample_folder)
img_path = os.path.join(sample_folder, random.choice(sample_images))

img = imread(img_path)
plt.imshow(img)
plt.title("Sample DR image")
plt.axis("off")
plt.show()
```


    
![png](DR_ML_files/DR_ML_26_0.png)
    


### Setting up the Generator


```python
# Setting up ImageDataGenerator for train/val/test
train_path = '/content/processed/train'
val_path = '/content/processed/val'
test_path = '/content/processed/test'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32
patience = 5 # number of epochs to wait to adjust if value does not improvee
stop_patience = 7  # number of epochs to wait before stopping the training if the value does not improve
epochs = 30

# Set up the training data generator using flow_from_directory
train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    train_path,
    target_size=img_size,
    class_mode='categorical', # Use 'categorical' for binary classification with 2 classes
    shuffle = True,
    batch_size=batch_size
)

# Setting up the validation data generator using flow_from_directory
val_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    val_path,
    target_size=img_size,
    class_mode='categorical',
    shuffle = True,
    batch_size=batch_size
)

# Seting up the test data generator using flow_from_directory
test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    test_path,
    target_size=img_size,
    class_mode='categorical',
    shuffle = False,
    batch_size=batch_size
)

print("Data generators train_batches, val_batches, and test_batches are now defined.")
```

    Found 2562 images belonging to 2 classes.
    Found 550 images belonging to 2 classes.
    Found 550 images belonging to 2 classes.
    Data generators train_batches, val_batches, and test_batches are now defined.



```python
# # Function to save the test images to google drive that later can be used for the streamlit app
# test_folder = '/content/processed/train'
# destination_test_folder = "/content/drive/MyDrive/diabetic_retinopathy_images/test"

# os.makedirs(destination_test_folder, exist_ok=True)
# shutil.copytree(test_folder, destination_test_folder, dirs_exist_ok=True)

# print("Images copied to Google Drive.")
```

    Images copied to Google Drive.


### Plotting the Images for Test


```python
# Image Batching
images, labels = next(train_batches)

# Class names from the generator
class_names = list(train_batches.class_indices.keys())

# Ploting 20 images to review in the training set
num_images_to_show = min(20, len(images))
plt.figure(figsize=(15, 15))

for i in range(num_images_to_show):
    plt.subplot(5, 4, i + 1)
    plt.imshow(images[i])
    # Get the true label
    true_label_index = np.argmax(labels[i])
    true_label = class_names[true_label_index]
    plt.title(f"Label: {true_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
```


    
![png](DR_ML_files/DR_ML_31_0.png)
    


### Building the Model & Learn


```python
# Call backs
# callbacks = [MyCallback(model=model, patience = patience, stop_patience = stop_patience, epochs =epochs)]

# Building the model
model = tf.keras.Sequential([
    layers.Conv2D(8, (3,3), padding="valid", input_shape=(224,224,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),

    layers.Conv2D(16, (3,3), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),

    layers.Conv2D(32, (4,4), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(32, activation = 'relu'),
    layers.Dropout(0.15),
    layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['acc'])

history = model.fit(train_batches, epochs=epochs, validation_data = val_batches)
```

    Epoch 1/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 179ms/step - acc: 0.8264 - loss: 0.4719 - val_acc: 0.5073 - val_loss: 0.6909
    Epoch 2/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 103ms/step - acc: 0.8889 - loss: 0.2746 - val_acc: 0.5073 - val_loss: 0.7208
    Epoch 3/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 114ms/step - acc: 0.9076 - loss: 0.2445 - val_acc: 0.5073 - val_loss: 0.7144
    Epoch 4/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 118ms/step - acc: 0.9187 - loss: 0.2292 - val_acc: 0.5855 - val_loss: 0.6698
    Epoch 5/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 101ms/step - acc: 0.9269 - loss: 0.2102 - val_acc: 0.7036 - val_loss: 0.4573
    Epoch 6/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 129ms/step - acc: 0.9217 - loss: 0.2173 - val_acc: 0.9182 - val_loss: 0.2779
    Epoch 7/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 120ms/step - acc: 0.9341 - loss: 0.1890 - val_acc: 0.9255 - val_loss: 0.2170
    Epoch 8/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 101ms/step - acc: 0.9327 - loss: 0.1908 - val_acc: 0.9309 - val_loss: 0.1983
    Epoch 9/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 114ms/step - acc: 0.9344 - loss: 0.1983 - val_acc: 0.9291 - val_loss: 0.1848
    Epoch 10/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 117ms/step - acc: 0.9423 - loss: 0.1597 - val_acc: 0.9382 - val_loss: 0.1887
    Epoch 11/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 99ms/step - acc: 0.9514 - loss: 0.1571 - val_acc: 0.9382 - val_loss: 0.1794
    Epoch 12/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 114ms/step - acc: 0.9481 - loss: 0.1586 - val_acc: 0.9400 - val_loss: 0.1800
    Epoch 13/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 115ms/step - acc: 0.9466 - loss: 0.1600 - val_acc: 0.9382 - val_loss: 0.1827
    Epoch 14/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 98ms/step - acc: 0.9503 - loss: 0.1427 - val_acc: 0.9436 - val_loss: 0.1762
    Epoch 15/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 110ms/step - acc: 0.9540 - loss: 0.1437 - val_acc: 0.9400 - val_loss: 0.1755
    Epoch 16/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 113ms/step - acc: 0.9533 - loss: 0.1503 - val_acc: 0.9364 - val_loss: 0.1753
    Epoch 17/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 99ms/step - acc: 0.9519 - loss: 0.1436 - val_acc: 0.9400 - val_loss: 0.1716
    Epoch 18/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 113ms/step - acc: 0.9576 - loss: 0.1273 - val_acc: 0.9400 - val_loss: 0.1716
    Epoch 19/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 114ms/step - acc: 0.9598 - loss: 0.1279 - val_acc: 0.9418 - val_loss: 0.1702
    Epoch 20/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 100ms/step - acc: 0.9616 - loss: 0.1245 - val_acc: 0.9418 - val_loss: 0.1749
    Epoch 21/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 111ms/step - acc: 0.9745 - loss: 0.0982 - val_acc: 0.9400 - val_loss: 0.1739
    Epoch 22/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 102ms/step - acc: 0.9687 - loss: 0.1048 - val_acc: 0.9436 - val_loss: 0.1692
    Epoch 23/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 98ms/step - acc: 0.9693 - loss: 0.1098 - val_acc: 0.9436 - val_loss: 0.1703
    Epoch 24/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 112ms/step - acc: 0.9691 - loss: 0.1027 - val_acc: 0.9382 - val_loss: 0.1732
    Epoch 25/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 111ms/step - acc: 0.9691 - loss: 0.1010 - val_acc: 0.9400 - val_loss: 0.1735
    Epoch 26/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 102ms/step - acc: 0.9722 - loss: 0.0926 - val_acc: 0.9473 - val_loss: 0.1651
    Epoch 27/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 113ms/step - acc: 0.9688 - loss: 0.0959 - val_acc: 0.9436 - val_loss: 0.1692
    Epoch 28/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 103ms/step - acc: 0.9765 - loss: 0.0900 - val_acc: 0.9473 - val_loss: 0.1639
    Epoch 29/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 108ms/step - acc: 0.9721 - loss: 0.0903 - val_acc: 0.9455 - val_loss: 0.1665
    Epoch 30/30
    [1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 111ms/step - acc: 0.9722 - loss: 0.1016 - val_acc: 0.9418 - val_loss: 0.1670


### Evaluating the Model


```python
# Evaluate the model on the test set
print("Evaluating model on the test set:")
loss, acc = model.evaluate(test_batches, verbose=1)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")
```

    Evaluating model on the test set:
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 75ms/step - acc: 0.9419 - loss: 0.1822
    Test Loss: 0.1910
    Test Accuracy: 0.9400


### Creating Confusion Matrix




```python
# Getting the true labels from the test_batches generator
test_batches.reset()
true_labels = test_batches.classes
class_names = list(test_batches.class_indices.keys())

# Get the model's predictions on the test data
predictions = model.predict(test_batches, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Classification report
report = classification_report(true_labels, predicted_classes, target_names=class_names)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)
```

    Confusion Matrix:
    [[259  20]
     [ 13 258]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
              DR       0.95      0.93      0.94       279
           No_DR       0.93      0.95      0.94       271
    
        accuracy                           0.94       550
       macro avg       0.94      0.94      0.94       550
    weighted avg       0.94      0.94      0.94       550
    


### Plotting the Confusion Matrix


```python
# Plotting the confusion Matrix for better overview

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```


    
![png](DR_ML_files/DR_ML_39_0.png)
    


**Reasoning**:
The confusion matrix and classification report were not successfully defined in the previous attempt.



### Saving the Model
Saving the Model as .keras


```python
# Define path to Google Drive folder
drive_folder_path = '/content/drive/MyDrive/trained_models'

# Create folder if it doesn't exist
os.makedirs(drive_folder_path, exist_ok=True)

# Define destination path
destination_path = os.path.join(drive_folder_path, 'CNN_model.keras')

# Save the model directly to Google Drive
try:
    model.save(destination_path)
    print(f"Model successfully saved to {destination_path}")
except Exception as e:
    print(f"An error occurred: {e}")

```

    Model successfully saved to /content/drive/MyDrive/trained_models/CNN_model.keras



```python
# Displaying Model Pereformance
train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
epochs = [i+1 for i in range(len(train_acc))]
loss_label = f"Best Epoch = {str(index_loss+1)}"
acc_label = f"Best Epoch = {str(index_acc + 1)}"

# Plotting Training History
plt.figure(figsize = (20,8))

plt.subplot(1,2,1)
plt.plot(epochs, train_loss, 'b', label = "Training Loss")
plt.plot(epochs, val_loss, 'g', label= "Validation Loss")
plt.scatter(index_loss + 1, val_lowest, s=150, c= "red", label = loss_label)
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, 'b', label = "Training Accuracy")
plt.plot(epochs, val_acc, 'g', label= "Validation Accuracy")
plt.scatter(index_acc + 1, acc_highest, s=150, c= "red", label = acc_label)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](DR_ML_files/DR_ML_43_0.png)
    


### Diabetes Retinopathy Detection

### Prediction on all images


```python
def predict_on_all_images_in_dir(directory_path, model):
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    if not image_files:
        print(f"Error: No image files found in {directory_path}")
        return

    print(f"Predicting images in: {directory_path}")
    true_class = os.path.basename(directory_path)
    print(f"True Class for this directory is: {true_class}")

    correct_predictions = 0
    total_images = len(image_files)

    for image_file in image_files:
        path = os.path.join(directory_path, image_file)

        img = cv2.imread(path)

        if img is None:
            print(f"Warning: I can'tload image from {path}. Please check.")
            continue

        RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        RGBimg = cv2.resize(RGBimg, (224, 224))

        image = np.array(RGBimg) / 255.0
        prediction = model.predict(np.array([image]))
        per = np.argmax(prediction, axis=1)[0]

        predicted_class = "DR" if per == 0 else "No_DR"

        print(f"  Image: {image_file}, Predicted Class: {predicted_class}")

        if true_class == predicted_class:
            correct_predictions += 1

    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f"\nSummary for {directory_path}:")
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

# Directories to test out
test_no_dr_dir = "/content/processed/test/No_DR"
test_dr_dir = "/content/processed/test/DR"

# Predicting on images in the No_DR directory
predict_on_all_images_in_dir(test_no_dr_dir, model)

print("-" * 30) # Separator

# Predict on images in the DR directory
predict_on_all_images_in_dir(test_dr_dir, model)
```

    Predicting images in: /content/processed/test/No_DR
    True Class for this directory is: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 782ms/step
      Image: 2a4520f1f9a3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 5ba156a35ff2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 7828dd083cdc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 98d41bce73a8.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 2b3a4a81d748.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 3325b1fe55d2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 77a1f1398fdb.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 436e7a7af761.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: bb85097857fa.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 44976c3b11a6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: b2b7ccd34cbd.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ef4121e9bb67.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: d7ab5c040294.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 51af8a689511.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ff03f74667df.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: ca05f7e7801b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 1faf8664816c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 91b7a4179ecf.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: baaca2f7e1f0.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
      Image: c23ff6dcf15e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 09c8323c612e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
      Image: 6b91e99c9408.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: e60e4edb3ca9.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
      Image: 6d4f6c9a8406.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
      Image: f90f8931a9bc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: f9d8ff3e6592.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: e06cccc08c59.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step
      Image: 874f8c1929f6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: ec0c9f817b03.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 4cddfc22b0ad.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
      Image: 7526c59c36d3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: bffca6eeb2bf.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: dee31065f8fe.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 2cdcc910778d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: fa6f3d8bb1d5.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
      Image: d99dd99be001.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step
      Image: 253e96488cfb.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step
      Image: f2f569a64949.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 50ms/step
      Image: 42b9c1977681.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 0fe31196e0e8.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 8bf2d925dc0c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 63c0eafd6aa9.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 3dec415b188a.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: d7bc00091cfc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: f066db7a2efe.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: 3a643599f852.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ac81fc200162.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 248dec89b3a2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: c5a0e84e955d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 4958bfcc9f38.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: fa0c87bd75ce.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
      Image: ace287a5c991.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 65e6f1bd9875.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ffd97f8cd5aa.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: e3cd96cb094c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 976082127e2a.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 35777eb7859d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 155e2df6bfcf.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a3132c8828e4.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 7da558d92100.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 835b9f6e12ba.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step
      Image: d667af5742f6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: fa9bece586fc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 79cbae28d8b2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 0e94cd271c00.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 38c7153457e2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 66460ecab347.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: 81d79d53ed7b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: ae57c8630249.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 86410aa13b3e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: d1ca85af57c9.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 60edda7b4871.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 31360e44ac64.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: c56293f53191.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: f58f0b2fd718.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
      Image: 36e4b704b905.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 6504b703c429.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: c755a0c4edcc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 3079490a4b9c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 19545647508e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: e7b5dd5bab1f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: a384e688e228.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: b9fe7da14a32.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
      Image: c334f8688b77.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 525d0dd8dc45.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 9ac2e3e9fca5.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: f57cf3b6f48e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 276b14f72328.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 84b88e8d3bca.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 973b0facfa9b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 946545473380.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 5fcff7280019.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step
      Image: 8329e80c10ac.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 27f82ada84ac.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 7131bf4c9e6f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 2c827005b8f8.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 54bdcdecd8f3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 63c7b0265775.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 0cecc2864b7f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: bf7b4eae7ad0.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: d30d079e6f9a.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 5ead17e894ae.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step
      Image: d56d32a1d62d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: fc898dfeb24f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 568455854a11.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ee3f5cf52188.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 6c9c902a97de.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 9d1feed37610.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 8e67f2d7e0ee.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 883c6a184f16.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 0da321efbce6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: b72a86d61959.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step
      Image: a8b3c0961d42.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 7a9f45fdf29b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: d4583e9525dc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: a0445785e2f7.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: d02b79fc3200.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 15b21c80cc31.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 9039cbfcbb2f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 0423237770a7.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 5a179c123fd8.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 9ac41b9a809e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 810ed108f5b7.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: b70e7c26f51e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: b7bd4a6627b6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: d332d7b8a26e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 51aa3361294c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: c1437a7a52c9.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a8dea22ef903.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: b0f0fa677d5f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 1002f3fe38f0.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 9bd008aab548.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a188c60b93fb.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: f62b8a076833.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 35df2bc6ae95.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 94076a9fb9b5.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 529906ff9dfa.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 8e2a3978c244.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: f66c4ee86629.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 55968f0e63c4.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 188219f2d9c6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step
      Image: a86128b601a7.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: aeccef0bdc26.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 07e827469099.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 5e97cb2b0888.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 8c84e96d9b01.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
      Image: 7d0a871c45db.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 80f6b30ece8c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 5889a0c75cac.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a73d012c4c38.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 1b398c0494d1.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: e79e10907295.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: e12f9f19d1be.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: c5bec7f1e5f3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 09662e462531.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 3ce2f8a77a32.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 2d3f4094c08a.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 565f3404f9b2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: ca2b54b95ade.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 96b5474ae604.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: b5b913358b32.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: a07d9a5045cb.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 49e4b95ee2dc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: 05b1bb2bdb81.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: 10a5026eb8e6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 5b47043942f4.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
      Image: e29e54ff921e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
      Image: 93421787f520.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: 3b2b91590590.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step
      Image: fb1b8771c70a.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: 7a06ea127e02.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 4c6c5a1bf5ab.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 43f22d1be8dd.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: 6b66b0e86f7e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 4dd14c380696.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 150fc7127582.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
      Image: 0d0b8fc9ab5c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: ab653b8554c0.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: e9678824215d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: 6c30dd481717.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: e69b48516577.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: 0232dfea7547.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: e9ce5bf645ab.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 50ms/step
      Image: 18cde9649e90.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
      Image: e03a74e7d74f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 22d843b2bbd1.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: 75ed83dbccce.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: 81914ceb4e74.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 4409965eb2a4.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 598b8f5b3822.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 359bab5d784b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: a1faeb4d5f10.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: ee77763a6afb.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 02cd34a85b24.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: cadde4030858.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
      Image: 5b32ece9c627.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: c26f98f58350.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 69c4cbb630de.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a8aed92940fb.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 96793edb1003.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: a1e236fbc863.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 274f4de2a59d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 5d5b5da5f939.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: cd01672507c9.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: da6389d129aa.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: de4cdabbce6d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: d2afca74cbc3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 4a589edaea60.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: cb68fce07789.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step
      Image: cd8da43e3069.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 67ed8cc78b97.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: aa94cc4bfd84.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 6f689fced922.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 174db0854291.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 883ddb650967.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: b0acd3593310.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: b82dfa63a75f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 7a42443ed106.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step
      Image: 191cf5668f33.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 0b8bdec9d869.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: 9122b31414d3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: bc92a61a1f9c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 461fa5292fda.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 6f3b62e5b7f5.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 47536db39f00.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 912fbe06407e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: b16dd4483ca5.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 81704925f759.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step
      Image: cf8ae5501bd6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: ec363f48867b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: ef7eb85b75fc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 68ddb15a74de.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 1ca62b3e4fd3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: bab776139279.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 3823acc4e464.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: f00ce9b9d6f4.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 1ca91751be4d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 7f43becd3e83.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 5995321563b7.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: ab32db41c409.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
      Image: 0924cec998fa.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: dfea19863428.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 54b322c66d01.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 53327edb9e4d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: 6852f4531591.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: b9c7c5182075.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 8b079e79035f.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 9232dc06cfdc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 7a12f49e29df.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 799214e8b07c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 83038ca49b6d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a53d6d2472a6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step
      Image: cb02bb47fdc5.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
      Image: d6130f2ec903.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: a1edf0e66592.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: f02956bd7c50.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 6d10709053ae.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 7eee3d1f1268.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 2974c6ad1d58.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 1fddd7c98fd2.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 9ce46d400cd6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ae20112e7a1e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: e82232a3c28b.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 9b4fc15df3c8.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 7b9d519cbd66.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 01d9477b1171.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: a02dfd67a925.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 50915e2329a1.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 5ac7a414560e.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: ec01f0862669.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 4276b82e4489.png, Predicted Class: No_DR
    
    Summary for /content/processed/test/No_DR:
    Total Images: 271
    Correct Predictions: 258
    Accuracy: 95.20%
    ------------------------------
    Predicting images in: /content/processed/test/DR
    True Class for this directory is: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 1e4b3b823b95.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a4d4b69f7404.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 4661006f3ba6.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 92d9e9f08709.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 5188a8afa879.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ed3a0fc5b546.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: b0d35981708b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 8273fdb4405e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ac720570dd0f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 1db18bdd43aa.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 916ec976ff30.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: d1cad012a254.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: f6f433f3306f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step
      Image: 1f07dae3cadb.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: f72ef9ceeaa8.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: c2d2b4f536da.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 6c3745a222da.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 8a67f1efa315.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 31cb39681f6a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 2665f72e2dd3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 18d8fdb140b7.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 69fff98cb32a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 5f70ad48a525.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 6531070bf03c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 38e111cac46f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 1e143fa3de57.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 110ms/step
      Image: aad0c0ee9268.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: b2ffa3e18559.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: 8c87bd748996.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 86fbac86ed3e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step
      Image: 0fd16b64697e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 36b5b3c9fb32.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
      Image: 5c8482926a08.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 0024cdab0c1e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step
      Image: 25d069089c5e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 56e56aa08362.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
      Image: 61bbc11fe503.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: a04fb36db784.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: 69591ebb198d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step
      Image: 54cab3596214.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: dd110d2b8c21.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
      Image: c5431b81cbc9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
      Image: 3a61e690f4bb.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: 65e120143825.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: 6ea07d19b4ce.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step
      Image: 465c618f7b23.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step
      Image: e32dc722eca5.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step
      Image: 79be2ff796bf.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 50ms/step
      Image: 7f84284598f5.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: 2463bb04ebc3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step
      Image: ffa47f6a7bf4.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step
      Image: 8ac0c44bbf24.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step
      Image: d10d315f123f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step
      Image: c73c5f6ef664.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 753b14c27c83.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: a4b8de38eac1.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 7bc2e0fa3f72.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 849a91e9ab28.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step
      Image: 0d310aba6373.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 115e42dd6a81.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 358d2224de73.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: de16416220de.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: db690e2d02f8.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: b3819a805dca.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 9eaf735cf01f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 188a9323be03.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: b2aaa81cc8f0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 232549883508.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 6810410187a0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 3ffa14d60b24.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 39aa3cd93c50.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: 4fa26d065ad3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 9c088d2d1559.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 175dd560810a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 6e092b306fe1.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 222f3ee3a1e8.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: e7a7187066ad.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step
      Image: 962cf85e4f6d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 247e98aba610.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 50840c36f0b4.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 1c3a6b4449e9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: f3a88d3026dc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 1e4650743fa2.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: e387311a840e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 191a711852bd.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 3dbfbc11e105.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 60f15dd68d30.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
      Image: 6a244e855d0e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: a3706ce27869.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: c52bb7343387.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 4478b870e549.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step
      Image: 513b0a4651fa.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 79540be95177.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: bfdee9be1f1d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 7e0598cc88a0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 8f318a978844.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 5f13e8a07344.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 0684311afdfc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: 8543a801dce0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step
      Image: b5c80d0ed0ff.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 33778d136069.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: d0079cc188e9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 31616ff6b53b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 52230bbef30e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 3de8ad4151e1.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 913490237ad4.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 5a2c27b95c7c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: a150ff5dfe07.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: d9bbdc33db83.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: bfd5c0e55420.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 2d552318eb07.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: bf9cba745efc.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 172df1330a60.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: d868acdccb5b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 9782c0489eca.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 40527a5e95dd.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 23fca0693e2a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: e019b3e0f33d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: ee74c3b177e0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 917f76f360b6.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: df84e7113003.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: f61bf44c677c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: b376def52ccc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: ababe19ed448.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 547b37da9223.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 38055d8b9f08.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step
      Image: e52ed5c29c5e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
      Image: 6b128e648646.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: e0b5a982a018.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 94ef1d14597f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 6c6efb6b1358.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: c96f743915b5.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 11b5c77fbf79.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 1a90fad9ffa2.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 49c5e7f6b8d2.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 157d17349cc6.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: eed4afc8ec83.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 9df31421cdd2.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 3a6e9730b298.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: 365f8c01d994.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: c406325360b1.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 756b0d6488bb.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step
      Image: 7a238a1d3cf3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: b49b2fac2514.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: f460608cf4cc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: f3b6b7ca1eb1.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 8bed09514c3b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: ed246ae1ed08.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 5347b4c8e9b3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 57a5f1015504.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: a77dbec966d4.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 9b418ce42c13.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 2376e5415458.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 87ms/step
      Image: 8596a24a14bd.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: 8f1e7433a95d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
      Image: 9dab2e6ba44b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: 5eb311bcb5f9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 7b29e3783919.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step
      Image: 99ecdb41d5e7.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step
      Image: 7455e2b5fc57.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
      Image: 7e4019ac7f5a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
      Image: 4462fba1d2a1.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: c7e827fc7f41.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step
      Image: 2f42e20db938.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step
      Image: e65f94ad9be3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: a7b7dc8788b9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 803120c5d287.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step
      Image: 7877be80901c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: 4d009cebabc9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step
      Image: 63363410389a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: a688f20f8895.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step
      Image: a3ad6c2db6f1.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 262ad704319c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step
      Image: 78a577c3e0bf.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
      Image: 910bfd38e2f5.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step
      Image: 9b95d6203406.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step
      Image: 62ab144d5cee.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
      Image: 4e6071b73120.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step
      Image: aa0afc41ed19.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: c4e8b1ec8893.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 19ef4d292196.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: d88806d9ece9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: bcd503c726ba.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 2cbfc6182ba2.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 8af6a4e5396f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: e7fc93ac5b6d.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 5efa24b03d5e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 613028ede6a0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: 8d3d67661620.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 4393c5bc576a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: bd269a1f0e4d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: d6283ded6aea.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 8ef2eb8c51c4.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 4a0890b08532.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 04d029cfb612.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: b6a0e348a01e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
      Image: a3fcf42ff56d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 5633ced07d8e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: bb9a3d835a94.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: a8582e346df0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 4d1cf360b2d7.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 274f5029189b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: a443c4fd489c.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: dc3c0d8ee20b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: b91ef82e723a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 2735be026d44.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
      Image: e4f12411fd85.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: f9ecf1795804.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: f9aa35187bf3.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
      Image: 1b32e1d775ea.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 96c3e3db68bc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: b762c29cf2f3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: c80f79579fed.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 71a39c660432.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: fc4c2d35c6f8.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: 92d8a7c8e718.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 9ad92f1c1542.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
      Image: ea4dcb055139.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 18323d8f2470.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 36677b70b1ef.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 1a03a7970337.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 07419eddd6be.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 3ddb86eb530e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 876e1dd12d38.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 959dc602febc.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 06b71823f9cd.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 23d7ca170bdb.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 9bf060db8376.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 51d0034d177d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 62ecdc90dd42.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: f8fc411092c7.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step
      Image: abf0f56c6f12.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: adb56cecafaf.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 5548a7961a3e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step
      Image: 3dfc50108072.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: fdd534271f3d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 4fecf87184e6.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 3c726de3ee90.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 388f12e8df0b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 224bb938e2dd.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 6cb98da77e3e.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 1124ffcd76c2.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: bfaa0080ab61.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 6630f8675a97.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step
      Image: 64a13949e879.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 50ms/step
      Image: ebe0175e530c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: a4d41c495666.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: a7ec056502e7.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step
      Image: 27bab1432f61.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
      Image: 7356dd08b0ae.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step
      Image: d28bd830c171.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 29580bed2f7d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step
      Image: d8404680bba6.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: b402b18d99a5.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: 20d5fdd450ae.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 383e72af1955.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 31b5d6fb0256.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 530d78467615.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step
      Image: 3f73c91b7e32.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
      Image: 0a61bddab956.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step
      Image: 3810040096cb.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
      Image: 82bb8a01935f.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: bba38f2294a3.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
      Image: 95a4cc805c7b.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: ed88faaa325a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
      Image: e724866f5084.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 0a9ec1e99ce4.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step
      Image: 91e8af9ceee9.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 61c2fbd16e38.png, Predicted Class: No_DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: cd5714db652d.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: c1c8550508e0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: e06d3d4733f0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
      Image: 3218a6d8eb2c.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step
      Image: 7663aba8d762.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step
      Image: c9d42d7534e0.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: 4e7694eebb91.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step
      Image: ca63fe4f4b52.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step
      Image: dee687c6e88a.png, Predicted Class: DR
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step
      Image: b640e3bdff75.png, Predicted Class: DR
    
    Summary for /content/processed/test/DR:
    Total Images: 279
    Correct Predictions: 259
    Accuracy: 92.83%


### Predicting With a Random Image


```python
def predict_class_random_from_dir(directory_path):
  if not os.path.exists(directory_path):
    print(f"Error: Directory not found at {directory_path}")
    return

  image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  if not image_files:
    print(f"Error: No image files found in {directory_path}")
    return

  # Randomly select an image in the file
  selected_image_file = random.choice(image_files)
  path = os.path.join(directory_path, selected_image_file)
  print(f"Predicting on random image: {selected_image_file}")

  # Extract the true class from the directory path
  true_class = os.path.basename(directory_path)
  print(f"True Class: {true_class}")

  img = cv2.imread(path)

  if img is None:
    print(f"Error: Could not load image from {path}. Check file integrity.")
    return

  RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  RGBimg = cv2.resize(RGBimg, (224, 224))
  plt.imshow(RGBimg)
  plt.axis("off")
  plt.show()

  image = np.array(RGBimg) / 255.0
  # Using the trained model
  prediction = model.predict(np.array([image]))
  per=np.argmax(prediction, axis=1)[0]

  # Corrected logic to match the true_class string format
  predicted_class = "DR" if per == 0 else "No_DR"
  print(f"Predicted Class: {predicted_class}")

  # Compare true and predicted classes
  if true_class == predicted_class:
      print("Prediction is CORRECT!")
  else:
      print("Prediction is INCORRECT.")

# List of directories to choose from & randomly select and predict from an image
test_directories = ["/content/processed/test/No_DR", "/content/processed/test/DR"]
selected_directory = random.choice(test_directories)
predict_class_random_from_dir(selected_directory)
```

    Predicting on random image: b3819a805dca.png
    True Class: DR



    
![png](DR_ML_files/DR_ML_48_1.png)
    


    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step
    Predicted Class: DR
    Prediction is CORRECT!

