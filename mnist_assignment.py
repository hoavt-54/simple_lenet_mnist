import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data

"""
LetNet-5
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

The code in this submission implements a simple convolutional neural network containing 2 conv layers 
which then followed by a fully connected layer. Each conv layer consists of a conv op and an pooling op.
I use maxpooling as pooling ops in this assignment. I also use TF-slim which reduce time of defining weights and biases. 

This implementation differs from the original architecture as well as added some modern findings in deep learning
1. making use of dropout
2. Relu instead of sigmoid
3. training by cross-entropy and AdamOptimizer instead of log-likelihood(eq Mean Square Error) and 


"""



def lenet(X, is_training=True):
    """
    X: a placeholder containing image data, shape [None, 28, 28, 1]
    """
    #reshape X as it is flat array in mnist dataset
    X = tf.reshape(X, [-1, 28,28,1])
    #first conv layer, same padding by default
    conv1 = slim.conv2d(X, 6, [3, 3], scope='conv1') #shape [None, 28, 28, 6]
    mp1 = slim.max_pool2d(conv1, [2, 2], scope='pool1') #shape [None, 14, 14, 6]
    #second conv layer
    conv2 = slim.conv2d(mp1, 16, [3, 3], scope='conv2') #shape [None, 28, 28, 16]
    mp2 = slim.max_pool2d(conv2, [2, 2], scope='pool2') #shape [None, 7, 7, 16]
    #last fully connected layer
    mp2_flatten = tf.contrib.layers.flatten(mp2)
    fc1 = slim.fully_connected(mp2_flatten, 120, scope='fc1') #shape[None, 120]
    drp = slim.dropout(fc1, 0.6, is_training=is_training, scope='dropout1')
    fc2 = slim.fully_connected(fc1, 84, scope='fc2') #shape[None, 84]
    drp = slim.dropout(fc2, 0.6, is_training=is_training, scope='dropout2')
    #output layer, no activation fn because there will be a softmax later when calculaing the loss
    out = slim.fully_connected(drp, 10, activation_fn=None, scope='fc3') #shape[None, 10]
    return out 

def train(mnist, lr=0.0001, num_epochs=8000, train_log=True):
    """
    Training using AdamOptimizer
    """
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    is_training = tf.placeholder(tf.bool, name='is_training')
    out = lenet(x, is_training)
    # Define the loss, optimizer and train op 
    loss = slim.losses.softmax_cross_entropy(out, y)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        #Run the initialization
        sess.run(init)
        for epoch in range (num_epochs):
            batch = mnist.train.next_batch(50)
            if train_log and epoch % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={ x: batch[0], y: batch[1], is_training:False}) 
                print('Training accuracy', train_accuracy)
            train_step.run(feed_dict={x: batch[0], y: batch[1], is_training:True})
        print(1 - accuracy.eval(feed_dict={
            x: mnist.test.images, y: mnist.test.labels, is_training:False}))

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print mnist.test.images.shape
    train(mnist)
