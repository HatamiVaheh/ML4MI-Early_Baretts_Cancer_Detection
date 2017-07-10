import tensorflow as tf
from keras.layers import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers.convolutional import Cropping2D
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.models import Model
from scipy import ndimage
from matplotlib import pyplot as plt
from keras.optimizers import Adam


imgData = ndimage.imread("image/sample image.jpg")

print(imgData.shape)


def visualizeImgData():
    plt.imshow(imgData)
    plt.show()

#visualizeImgData()


def get_Unet():
    inputs = Input((572, 572, 3))
    # Layer one - downsize
    conv1 = Conv2D(64,(3,3), strides=(1,1), activation='relu', input_shape=(572, 572, 3))(inputs)
    conv1 = Conv2D(64,(3,3), strides=(1,1), activation='relu')(conv1)

    pool1 = MaxPool2D((2,2), strides=2)(conv1)     # (284x284x64)

    # Layer two - downsize
    conv2 = Conv2D(128,(3,3), strides=(1,1), activation='relu')(pool1)
    conv2 = Conv2D(128,(3,3), strides=(1,1), activation='relu')(conv2)

    pool2 = MaxPool2D((2,2), strides=2)(conv2)     # (140x140x128)

    # Layer three - downsize
    conv3 = Conv2D(256,(3,3), strides=(1,1), activation='relu')(pool2)
    conv3 = Conv2D(256,(3,3), strides=(1,1), activation='relu')(conv3)

    pool3 = MaxPool2D((2,2), strides=2)(conv3)     # (68x68x256)

    # Layer four - downsize
    conv4 = Conv2D(512,(3,3), strides=(1,1), activation='relu')(pool3)
    conv4 = Conv2D(512,(3,3), strides=(1,1), activation='relu')(conv4)

    pool4 = MaxPool2D((2,2),strides=2)(conv4)     # (32x32x512)

    # Layer five - last layer of downsizing
    conv5 = Conv2D(1024,(3,3), strides=(1,1), activation='relu')(pool4)
    conv5 = Conv2D(1024,(3,3), strides=(1,1), activation='relu')(conv5)

    # crop conv4 layer
    conv4Crop= tf.image.resize_image_with_crop_or_pad(conv4, 56,56)
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2))(conv5), conv4Crop], axis=3)

    # Layer six - upsizing
    conv6 = Conv2D(512,(3,3), strides=(1,1), activation='relu')(up6)
    conv6 = Conv2D(512,(3,3), strides=(1,1), activation='relu')(conv6)

    # crop conv3 layer
    conv3Crop= tf.image.resize_image_with_crop_or_pad(conv3, 104,104)
    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2))(conv6), conv3Crop], axis=3)

    # Layer seven - upsizing
    conv7 = Conv2D(256,(3,3), strides=(1,1), activation='relu')(up7)
    conv7 = Conv2D(256,(3,3), strides=(1,1), activation='relu')(conv7)

    # crop conv2 layer
    conv2Crop= tf.image.resize_image_with_crop_or_pad(conv2, 200,200)
    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2))(conv7), conv2Crop], axis=3)

    # Layer eight - upsizing
    conv8 = Conv2D(128,(3,3), strides=(1,1), activation='relu')(up8)
    conv8 = Conv2D(128,(3,3), strides=(1,1), activation='relu')(conv8)

    # crop conv1 layer
    conv1Crop= tf.image.resize_image_with_crop_or_pad(conv1, 392,392)
    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8), conv1Crop], axis=3)

    # Layer nine - upsizing final layer
    conv9 = Conv2D(64,(3,3), strides=(1,1), activation='relu')(up9)
    conv9 = Conv2D(64,(3,3), strides=(1,1), activation='relu')(conv9)
    conv9 = Conv2D(2, (1,1), strides=(1,1))(conv9)

    model = Model(inputs=[inputs], outputs=[conv9])
    model.compile(optimizer=Adam(lr=1e-5), loss="mean_squared_error")
    return model



if __name__ == "__main__":
    pass
    #to train the model
    #get_Unet().fit()