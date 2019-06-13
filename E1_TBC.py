import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import rotate_mnist as rmn
pi = 3.141592

#get dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
x_train, x_test = x_train / 255.0, x_test / 255.0

#parameters
iterations = 1000
learn_rate = 0.0001
batch_size = 250
dataset_size = y_train.shape[0]
test_dataset_size = y_test.shape[0]
num_batches = dataset_size // batch_size
num_test_batches = test_dataset_size // batch_size
num_batches_per_it = num_test_batches
report_iter = 10
img_size = 32

#convolutional network with 3 conv layers, followed by a shared dense layer, and individual dense layers
#(maxpooling after first two conv layers)
def build_network(x, prob):
    conv1 = tf.layers.max_pooling2d(tf.layers.conv2d(tf.reshape(x,[-1,img_size,img_size,1]),32,5,activation=tf.nn.relu),2,2)
    conv2 = tf.layers.max_pooling2d(tf.layers.conv2d(conv1,128,5,activation=tf.nn.relu),2,2)
    conv3 = tf.layers.conv2d(conv2,256,3,activation=tf.nn.relu)
    dense1 = tf.layers.dense(tf.layers.dropout(tf.layers.flatten(conv3),prob),512,activation=tf.nn.relu)

    # predict output identity and angle with uncertainty estimate for the latter
    pred_y_logits = tf.layers.dense(tf.layers.dropout(dense1, prob), 10)
    pred_angle = tf.layers.dense(tf.layers.dropout(dense1, prob), 1, bias_initializer=tf.initializers.zeros())
    pred_angle = tf.reshape(pred_angle, [-1,])
    pred_angle_unc_logit = tf.layers.dense(tf.layers.dropout(dense1, prob), 1,
                                           bias_initializer=tf.initializers.constant(1))
    pred_angle_unc_logit = tf.reshape(pred_angle_unc_logit, [-1,])
    return pred_y_logits, pred_angle, pred_angle_unc_logit


def digit_loss(pred_logits, ground_truth):
    return tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=pred_logits)


#TODO: Gaussian loss with uncertainty
def gauss_loss(mean, log_sigma, val):
    return tf.zeros(shape=())



#build network
#inputs
x = tf.placeholder(tf.float32, [None, img_size, img_size], name="x")
y = tf.placeholder(tf.int32, [None,], name="y")
angle = tf.placeholder(tf.float32, [None,], name="angle")
prob = tf.placeholder_with_default(1.0, shape=(), name="prob")
pred_y_logits, pred_angle, pred_angle_unc_logit = build_network(x, prob)

#output in logits transform to actual units
pred_y = tf.math.argmax(pred_y_logits,axis=-1)
pred_angle_unc = tf.exp(pred_angle_unc_logit)

#compute losses and the network cost
loss_y = digit_loss(pred_y_logits,y)
loss_angle = gauss_loss(pred_angle,pred_angle_unc_logit,angle)
cost = loss_y + loss_angle
train = tf.train.AdamOptimizer(learn_rate).minimize(cost)

#some error metrics to watch while optimizing
error_angle = tf.math.sqrt(tf.math.reduce_mean(tf.math.square((pred_angle-angle))))
uncertainty_angle = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(pred_angle_unc)))

#initializer for network
init = tf.global_variables_initializer()

#prepare for visualization

with tf.Session() as session:

    #initalize network
    session.run(init)

    #train network
    for iter in range(iterations+1):
        acc_cost = 0
        acc_cost_y = 0
        acc_cost_angle = 0
        acc_err_angle = 0
        acc_unc_angle = 0

        #get randomized indices for this batch
        batch_index = np.random.permutation(dataset_size)

        for b in range(num_batches_per_it):

            #get data for this batch, then upsample and rotate
            this_batch_x = np.zeros([batch_size,img_size,img_size])
            this_batch_theta = np.float32(np.random.normal(size=(batch_size,)) * pi / 6)
            this_batch_theta = (this_batch_theta+pi) % (2*pi) - pi

            for k in range(batch_size):
                this_batch_x[k,:,:] = rmn.generate_image(x_train[batch_index[(b*batch_size)+k],:,:], this_batch_theta[k],img_size)

            #get desired output
            this_batch_y = y_train[batch_index[(b * batch_size):((b + 1) * batch_size)]]

            _, this_cost, this_loss_y, this_loss_angle, this_err_angle, this_unc_angle = \
                session.run([train,cost,loss_y,loss_angle,error_angle,uncertainty_angle], feed_dict={
                x: this_batch_x,
                y: this_batch_y,
                angle: this_batch_theta,
                prob: 0.25,
            })
            acc_cost += this_cost
            acc_cost_y += this_loss_y
            acc_cost_angle += this_loss_angle
            acc_err_angle += this_err_angle
            acc_unc_angle += this_unc_angle

        print(str(iter) + " Losses: " + str(acc_cost / num_batches) + "\tnumber " + str(acc_cost_y / num_batches) + "\t angle: " + str(acc_cost_angle / num_batches) + " "
              + str(acc_err_angle / num_batches) + " " + str(acc_unc_angle / num_batches), flush=True)

        # evaluate network
        if( iter % report_iter == 0):
            pred_y_log = np.zeros([test_dataset_size])
            error_angle_log = np.zeros([test_dataset_size])
            unc_angle_log = np.zeros([test_dataset_size])
            for b in range(num_test_batches):
                this_batch_theta = np.float32(np.random.normal(size=(batch_size,)) * pi / 6)
                for k in range(batch_size):
                    this_batch_x[k,:,:] = rmn.generate_image(x_test[(b*batch_size)+k,:,:], this_batch_theta[k],img_size)
                this_batch_y = y_test[b*batch_size:((b+1)*batch_size)]

                this_pred_y, this_pred_angle, this_unc_angle = \
                    session.run([pred_y, pred_angle, pred_angle_unc], feed_dict={
                    x: this_batch_x,
                    y: this_batch_y,
                    angle: this_batch_theta,
                    prob: 0.0,
                })
                for k in range(batch_size):
                    pred_y_log[b*batch_size+k] = this_pred_y[k]
                    error_angle_log[b*batch_size+k] = this_pred_angle[k]-this_batch_theta[k]
                    unc_angle_log[b*batch_size+k] = this_unc_angle[k]

            for a in range(10):
                fig = plt.figure(figsize=(60,45))
                fig.gca().scatter(error_angle_log[y_test[:]==a], unc_angle_log[y_test[:]==a])
                fig.gca().plot([-2, 0, 2],[2, 0, 2],'k--')
                fig.gca().plot([-2, 0, 2],[1, 0, 1],'r--')
                fig.gca().set_xlim(-pi/3,pi/3)
                fig.gca().set_ylim(0,pi/3)
                fig.gca().set_title("Iter " + str(iter) + " Digit #" + str(a))
                fig.savefig('Iter' + str(iter) + 'Digit' + str(a) + '.pdf')
                plt.close(fig)