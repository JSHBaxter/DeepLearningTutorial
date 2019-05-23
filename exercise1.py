import numpy as np
import cv2
import scipy.ndimage as spi
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#get dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
y_train = np.expand_dims(y_train, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

iterations = 50
learn_rate = 0.001
batch_size = 250
dataset_size = y_train.shape[0]
test_dataset_size = y_test.shape[0]
num_batches = dataset_size // batch_size
num_test_batches = test_dataset_size // batch_size
img_size = 32
pi = 3.141592

def generate_image(input_img, rotation):
    output_img = cv2.resize(input_img, dsize=(img_size,img_size))
    output_img = spi.rotate(output_img, 180 * rotation / pi, reshape=False)
    return output_img




x = tf.placeholder(tf.float32, [None, img_size, img_size], name="x")
y = tf.placeholder(tf.uint8, [None, 1], name="y")
angle = tf.placeholder(tf.float32, [None, 1], name="angle")

conv1 = tf.layers.max_pooling2d(tf.layers.conv2d(tf.reshape(x,[-1,img_size,img_size,1]),32,5,activation=tf.nn.relu),2,2)
conv2 = tf.layers.max_pooling2d(tf.layers.conv2d(conv1,128,5,activation=tf.nn.relu),2,2)
conv3 = tf.layers.conv2d(conv2,256,3,activation=tf.nn.relu)

dense1 = tf.layers.dense(tf.layers.dropout(tf.layers.flatten(conv3),rate=0.25),512,activation=tf.nn.relu)
pred_y_logits = tf.layers.dense(tf.layers.dropout(dense1,rate=0.25),10)
pred_y = tf.math.argmax(pred_y_logits,axis=-1)
pred_angle = tf.layers.dense(tf.layers.dropout(dense1,rate=0.25),1)
pred_angle_unc_logit = tf.layers.dense(tf.layers.dropout(dense1,rate=0.25),1)
pred_angle_unc = tf.exp(pred_angle_unc_logit)

loss_y = tf.losses.softmax_cross_entropy(tf.reshape(tf.one_hot(y,10),[-1,10]),pred_y_logits)
loss_angle = tf.math.reduce_mean(0.5*tf.math.square((pred_angle-angle)*tf.exp(-pred_angle_unc_logit)) + pred_angle_unc_logit)

error_angle = tf.math.sqrt(tf.math.reduce_mean(tf.math.square((pred_angle-angle))))
uncertainty_angle = tf.math.reduce_mean(tf.exp(pred_angle_unc))

cost = loss_y + loss_angle
train = tf.train.AdamOptimizer(learn_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:

    #initalize network
    session.run(init)

    #train network
    for iter in range(iterations):
        acc_cost = 0
        acc_cost_y = 0
        acc_cost_angle = 0
        acc_err_angle = 0
        acc_unc_angle = 0

        #get randomized indices for this batch
        batch_index = np.random.permutation(dataset_size)

        for b in range(num_batches):

            #get data for this batch, then upsample and rotate
            this_batch_x = np.zeros([batch_size,img_size,img_size])
            this_batch_theta = np.float32(np.random.normal(size=(batch_size,1)) * pi / 6)
            this_batch_theta = (this_batch_theta+pi) % (2*pi) - pi

            for k in range(batch_size):
                this_batch_x[k,:,:] = generate_image(x_train[batch_index[(b*batch_size)+k],:,:], this_batch_theta[k,:])

            #get desired output
            this_batch_y = y_train[batch_index[(b * batch_size):((b + 1) * batch_size)], :]

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

    #evaluate network
    pred_y_log = np.zeros([test_dataset_size])
    error_angle_log = np.zeros([test_dataset_size])
    unc_angle_log = np.zeros([test_dataset_size])
    for b in range(num_test_batches):
        this_batch_theta = np.float32(np.random.normal(size=(batch_size,1)) * pi / 6)
        for k in range(batch_size):
            this_batch_x[k,:,:] = generate_image(x_test[(b*batch_size)+k,:,:], this_batch_theta[k,:])
        this_batch_y = y_test[b*batch_size:((b+1)*batch_size), :]

        this_pred_y, this_pred_angle, this_unc_angle = \
            session.run([pred_y, pred_angle, pred_angle_unc], feed_dict={
            x: this_batch_x,
            y: this_batch_y,
            angle: this_batch_theta
        })
        for k in range(batch_size):
            pred_y_log[b*batch_size+k] = this_pred_y[k]
            error_angle_log[b*batch_size+k] = this_pred_angle[k,0]-this_batch_theta[k,0]
            unc_angle_log[b*batch_size+k] = this_unc_angle[k]

    for a in range(10):
        fig, ax = plt.subplots()
        ax.scatter(error_angle_log[y_test[:,0]==a], unc_angle_log[y_test[:,0]==a])
        ax.plot([-2, 0, 2],[2, 0, 2],'k--')
        ax.set_title("Digit #" + str(a))
        fig.show()

while True:
    time.sleep(0.1)