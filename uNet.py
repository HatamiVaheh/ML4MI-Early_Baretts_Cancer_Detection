import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.models import Sequential
from scipy import ndimage
from matplotlib import pyplot as plt


imgData = ndimage.imread("image/sample image.jpg")

print(imgData.shape)


def visualizeImgData():
    plt.imshow(imgData)
    plt.show()

#visualizeImgData()


model = Sequential()

# Layer one - downsize
model.add(Conv2D(64,(3,3), strides=(1,1), activation='relu', input_shape=(572, 572, 3)))
model.add(Conv2D(64,(3,3), strides=(1,1), activation='relu'))

model.add(MaxPool2D((2,2)))     # (284x284x64)

# Layer two - downsize
model.add(Conv2D(128,(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(128,(3,3), strides=(1,1), activation='relu'))

model.add(MaxPool2D((2,2)))     # (140x140x128)

# Layer three - downsize
model.add(Conv2D(256,(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(256,(3,3), strides=(1,1), activation='relu'))

model.add(MaxPool2D((2,2)))     # (68x68x256)

# Layer four - downsize
model.add(Conv2D(512,(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(512,(3,3), strides=(1,1), activation='relu'))

model.add(MaxPool2D((2,2)))     # (32x32x512)

# Layer five - last layer of downsizing
model.add(Conv2D(1024,(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(1024,(3,3), strides=(1,1), activation='relu'))

print(model.output_shape)