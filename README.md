# Handwritten-Numbers-Recognition
Here I use a convolutional neural network with the MNIST dataset to recognise handwritten numbers

This was my first neural network project. I use tensorflow as my neural network library, and imported the MNIST dataset with 60,000 train images and 10,000 test images. 

I used a CNN, wiht the LeNet-5 structure. The images are resized to be greyscale and 28x28. Then the first layer of the Neural Network is a 2D convolution with 6 filters and a kernel size of (5,5) and same padding. Then there is a max pooling layer. Following that is another convolution layer with 16 filters this time and no padding, followed by another max pooling layer. Then the image is flattened, and passed through 1 dense layer of size 120, then to the output layer. The network has 51,902 trainable parameters.

This gave an accuracy of 98.9%. 

I proceeded to write some functions that processed images that I entered into the code so that they could be resized. 
