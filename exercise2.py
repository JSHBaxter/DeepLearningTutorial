import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#generate image
def generate_image(size, snr):
    img = np.zeros((size,size),dtype='float32')
    idx_x, idx_y = np.meshgrid(range(size),range(size))
    idx_x, idx_y = idx_x / size, idx_y / size
    img[ idx_x >= idx_y ] = -1.0
    img[ idx_x < idx_y ] = 1.0
    img[ (idx_x-0.75)**2 + (idx_y-0.25)**2 < 0.2**2 ] = 0.0
    img[ (idx_x-0.25)**2 + (idx_y-0.75)**2 < 0.2**2 ] = 0.0
    return img+np.random.normal(size=img.shape) / snr

def generate_mask(size):
    img = np.zeros((size,size),dtype='int32')
    idx_x, idx_y = np.meshgrid(range(size),range(size))
    idx_x, idx_y = idx_x / size, idx_y / size
    img[ idx_x >= idx_y ] = 0
    img[ idx_x < idx_y ] = 1
    img[ (idx_x-0.75)**2 + (idx_y-0.25)**2 < 0.2**2 ] = 2
    img[ (idx_x-0.25)**2 + (idx_y-0.75)**2 < 0.2**2 ] = 3
    return img


#build database from images
def build_database(number,size,snr):
    database_i = np.zeros((number,size,size,1),dtype='float32')
    database_o = np.zeros((number,size,size),dtype='int32')
    for n in range(number):
        database_i[n,:,:,0] = generate_image(size,snr)
        database_o[n,:,:] = generate_mask(size)
    return database_i, database_o

#single iteration of a DenseCRF mean field approximation iteration
def DenseCRF_MeanApproxIteration(Q_in, P_in, k, weight):
    #perform message passing
    Q_unweighted = tf.conv2d(Q_in, k)
    Q_weighted = tf.conv2d(Q_unweighted, weight)
    Q_weighted = tf.conv2d(Q_weighted, compatibility)

    #add in unary potentials
    Q_unnormed = P_in - Q_weighted

    #normalize solution
    Q_out = tf.nn.softmax(Q)
    return Q_out

learn_rate = 0.01
iterations = 1000
img_size = 32
img_snr = 1
training_size = 25
testing_size = 25

training_i, training_o = build_database(training_size,img_size,img_snr)

#build network
#inputs
x = tf.placeholder(tf.float32, [None, img_size, img_size,1], name="x")
y = tf.placeholder(tf.int32, [None, img_size, img_size], name="y")

#network
conv1 = tf.layers.conv2d(x,32,3,padding='same',activation=tf.nn.relu)
conv2 = tf.layers.conv2d(conv1,128,1,padding='same',activation=tf.nn.relu)
conv3 = tf.layers.conv2d(conv2,256,1,padding='same',activation=tf.nn.relu)
conv4 = tf.layers.conv2d(conv2,4,1,padding='same')

#weights for CNN

#outputs
pred_logits = conv4
pred_y = tf.nn.softmax(pred_logits)
acc = 1.0- tf.math.reduce_mean(tf.math.abs(tf.one_hot(y,4)-pred_y))

#costs and initializer
cost =  tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=pred_logits))
train = tf.train.AdamOptimizer(learn_rate).minimize(cost)
init = tf.global_variables_initializer()


#initializer for network
init = tf.global_variables_initializer()

with tf.Session() as session:

    #initalize network
    session.run(init)

    #train network
    for iter in range(iterations+1):

        _, this_cost, this_acc = \
            session.run([train, cost, acc], feed_dict={
                x: training_i,
                y: training_o
            })

        print(str(iter) + " Losses: " + str(this_cost) + " " + str(this_acc))



while(True):
    time.sleep(0.01)