import tensorflow as tf
import keras
import keras.utils
from keras.utils import to_categorical
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import numpy as np
import cv2
(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

def preprocess_data(x_train, y_train, x_eval, y_eval):
    #reshape images
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
    #x_eval = x_train.reshape(x_eval.shape[0], x_eval.shape[1], x_eval.shape[2],1)

    #convert image values from integers to floats
    x_train = x_train.astype("float32")
    x_eval = x_eval.astype("float32")
    
    #normalising data
    x_train = x_train/255.0
    x_eval = x_eval/255.0

    #encoding labels
    y_train = to_categorical(y_train)
    y_eval = to_categorical(y_eval)

    return x_train, y_train, x_eval, y_eval

#this code is used to train the neural network
"""
model = Sequential()
model.add(Conv2D(filters = 6, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

# Rescaling all training and testing data
x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_eval, y_eval)
# Fitting the model on the training set
history = model.fit(x_train, y_train, epochs = 10, batch_size = 50,validation_split=0.2,shuffle=True)
_, acc = model.evaluate(x_test, y_test, verbose = 1)

model.save("MNIST NN save")
"""

model = keras.models.load_model("MNIST NN save")
x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_eval, y_eval)
_, acc = model.evaluate(x_test, y_test, verbose = 1)
model.summary()
def photo_editor(image):
    image = cv2.resize(image, (28,28))
    #image = cv2.bitwise_not(image)
    image = np.array(image)
    image = np.invert(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32")
    image /=255.0
    return image

image_input_path = ["./input - Copy/Predictions/Number 1.jpg",
                    "./input - Copy/Predictions/Number 2.jpg",
                    "./input - Copy/Predictions/Number 3.jpg",
                    "./input - Copy/Predictions/Number 4.jpg",
                    "./input - Copy/Predictions/Number 5.jpg",
                    "./input - Copy/Predictions/Number 6.jpg",
                    "./input - Copy/Predictions/Number 7.jpg",
                    "./input - Copy/Predictions/Number 8.jpg",
                    "./input - Copy/Predictions/Number 9.jpg",
                    "./input - Copy/Predictions/Number 16.jpg",
                    ]

fig,axs = plt.subplots(5,5,figsize=[24,21])
count = 0
for i in range(5):
    for j in range(5):
        try:
            image = cv2.imread(image_input_path[count], 0)
            img = photo_editor(image)
            
            vec_p = model.predict(img)
            result = np.argsort(vec_p)
            result = result[0][::-1]
            print(result)
            print(result, "Image:", count)
            axs[i][j].imshow(cv2.imread(image_input_path[count]))
            axs[i][j].set_title(str("Prediction: " + str(result[0])))

            count +=1
        except:
            image = cv2.imread(image_input_path[0], 0)
            img = photo_editor(image)
            
            vec_p = model.predict(img)
            result = np.argsort(vec_p)
            result = result[0][::-1]
            print(result, "Image:", count)
            axs[i][j].imshow(cv2.imread(image_input_path[0]))
            axs[i][j].set_title(str("Prediction: " + str(result[0])))

            count +=1

print(len(y_train))
print(len(y_eval))

plt.suptitle("Hope for the beset")
plt.show()
