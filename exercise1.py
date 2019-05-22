import numpy as np
import cv2
import scipy.ndimage as spi
import tensorflow as tf
import matplotlib.pyplot as plt

iterations = 10000
learn_rate = 0.01
batch_size = 1000
num_batches = 60000 // batch_size
img_size = 56

def generate_image(input_img, rotation):
    output_img = cv2.resize(input_img, dsize=(img_size,img_size))
    output_img = spi.rotate(output_img, 180 * rotation / 3.141592, reshape=False)
    return output_img




mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
y_train = np.expand_dims(y_train, axis=-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

x = tf.placeholder(tf.float32, [None, img_size, img_size], name="x")
y = tf.placeholder(tf.uint8, [None, 1], name="y")
angle = tf.placeholder(tf.float32, [None, 1], name="angle")

conv1 = tf.layers.max_pooling2d(tf.layers.conv2d(tf.reshape(x,[-1,img_size,img_size,1]),32,5,activation=tf.nn.relu),2,2)
conv2 = tf.layers.max_pooling2d(tf.layers.conv2d(conv1,128,5,activation=tf.nn.relu),2,2)
conv3 = tf.layers.conv2d(conv2,256,3,activation=tf.nn.relu)

dense1 = tf.layers.batch_normalization(tf.layers.dense(tf.layers.flatten(conv3),512,activation=tf.nn.relu))
pred_y = tf.layers.batch_normalization(tf.layers.dense(dense1,10))
pred_angle = tf.layers.batch_normalization(tf.layers.dense(dense1,1))
pred_angle_unc = tf.layers.batch_normalization(tf.layers.dense(dense1,1))

loss_y = tf.losses.softmax_cross_entropy(tf.reshape(tf.one_hot(y,10),[-1,10]),pred_y)
loss_angle = tf.math.reduce_mean(0.5*tf.math.square((pred_angle-angle)*tf.exp(-pred_angle_unc)) + pred_angle_unc)

error_angle = tf.math.sqrt(tf.math.reduce_mean(tf.math.square((pred_angle-angle))))
uncertainty_angle = tf.math.reduce_mean(tf.exp(pred_angle_unc))

cost = loss_y + loss_angle
train = tf.train.AdamOptimizer(learn_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for iter in range(iterations):
        acc_cost = 0
        acc_cost_y = 0
        acc_cost_angle = 0
        acc_err_angle = 0
        acc_unc_angle = 0

        for b in range(num_batches):

            #get data for this batch, then upsample and rotate
            this_batch_x = np.zeros([batch_size,img_size,img_size])
            this_batch_theta = np.float32(np.random.normal(size=(batch_size,1)) * 3.141592 / 6)
            pre_sized = x_train[(b*batch_size):((b+1)*batch_size),:,:]
            for k in range(batch_size):
                this_batch_x[k,:,:] = generate_image(pre_sized[k,:,:], 180 * this_batch_theta[i,:] / 3.141592)

            #get desired output
            this_batch_y = y_train[(b * batch_size):((b + 1) * batch_size), :]

            _, this_cost, this_loss_y, this_loss_angle, this_err_angle, this_unc_angle = \
                session.run([train,cost,loss_y,loss_angle,error_angle,uncertainty_angle], feed_dict={
                x: this_batch_x,
                y: this_batch_y,
                angle: this_batch_theta
            })
            acc_cost += this_cost
            acc_cost_y += this_loss_y
            acc_cost_angle += this_loss_angle
            acc_err_angle += this_err_angle
            acc_unc_angle += this_unc_angle

        print(str(iter) + " Losses: " + str(acc_cost / num_batches) + "\tnumber " + str(acc_cost_y / num_batches) + "\t angle: " + str(acc_cost_angle / num_batches) + " "
              + str(acc_err_angle / num_batches) + " " + str(acc_unc_angle / num_batches))


