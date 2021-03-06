import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import time
import rotate_mnist as rmn

pi = 3.141592

# get dataset
mnist = K.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
x_train, x_test = x_train / 255.0, x_test / 255.0

# parameters
iterations = 1000
learn_rate = 0.001
batch_size = 250
dataset_size = y_train.shape[0]
test_dataset_size = y_test.shape[0]
num_batches = dataset_size // batch_size
num_test_batches = test_dataset_size // batch_size
num_batches_per_it = num_batches
report_iter = 5
img_size = 32
style_dims = 2
viz_iter = 10

# convolutional network encoder with 3 conv layers, followed by a dense layer
# (maxpooling after first two conv layers)
def build_encoder(x,prob):
    convE1 = tf.layers.max_pooling2d(
        tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, name="convE1"), 2, 2)
    convE2 = tf.layers.max_pooling2d(tf.layers.conv2d(convE1, 128, 5, activation=tf.nn.relu, name="convE2"), 2, 2)
    convE3 = tf.layers.conv2d(convE2, 256, 3, activation=tf.nn.relu, name="convE3")
    denseE1 = tf.layers.dense(tf.layers.dropout(tf.layers.flatten(convE3), prob),
                              512, activation=tf.nn.relu, name="denseE1")

    # predict output identity and style parameters as internal state
    pred_y_logits = tf.layers.dense(tf.layers.dropout(denseE1, prob), 10, name="pred_y_logits")
    style_state_means = tf.layers.dense(tf.layers.dropout(denseE1, prob), style_dims,
                                        name="style_state_means", bias_initializer=tf.initializers.zeros())
    style_state_stds = tf.layers.dense(tf.layers.dropout(denseE1, prob), style_dims,
                                       name="style_state_stds", bias_initializer=tf.initializers.zeros())

    return pred_y_logits, style_state_means, style_state_stds, \
           ["convE1", "convE2", "convE3", "denseE1", "pred_y_logits", "style_state_means", "style_state_stds"]

def get_state(pred_y_logits, style_state_means, style_state_stds, training_flag):
    y_state = tf.nn.softmax(pred_y_logits, axis=-1)
    style_state = style_state_means + \
              training_flag * style_state_stds * tf.random.normal(tf.shape(style_state_stds))
    return tf.concat([y_state, style_state], axis=-1)


def build_decoder(state,prob):
    denseD1 = tf.layers.dense(state, 512, activation=tf.nn.relu, name='denseD1')
    denseD2 = tf.layers.dense(denseD1, int(img_size//4)*int(img_size//4)*32, activation=tf.nn.relu, name='denseD2')
    denseD2 = tf.reshape(denseD2,[-1,int(img_size//4),int(img_size//4),32])
    convD3 = tf.layers.conv2d_transpose(denseD2, 64, kernel_size=5, strides=[2, 2],
                                    padding="SAME", activation=tf.nn.relu, name='convD3')
    convD2 = tf.layers.conv2d_transpose(convD3,32,kernel_size=5, strides=[2, 2],
                                    padding="SAME", activation=tf.nn.relu, name='convD2')
    convD1 = tf.layers.conv2d(convD2,1,kernel_size=1, name='convD1')
    return convD1, \
           ["denseD1", "denseD2", "convD3", "convD2", "convD1"]


def digit_loss(pred_logits, ground_truth):
    return tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=pred_logits)


def kl_loss(means, stds):
    return tf.math.reduce_mean( tf.math.square(means) + tf.math.square(stds) - tf.math.log(tf.math.square(stds)) - 1)


def recon_loss(recon_x_logits, x):
    return tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=recon_x_logits))


# build network
# inputs
x = tf.placeholder(tf.float32, [None, img_size, img_size,1], name="x")
y = tf.placeholder(tf.int32, [None,], name="y")
training_flag = tf.placeholder_with_default(1.0, shape=(), name="training_flag")
prob = tf.placeholder_with_default(0.5, shape=(), name="prob")

pred_y_logits, style_state_means, style_state_stds, enc_layers = build_encoder(x, prob)
state = get_state(pred_y_logits, style_state_means, style_state_stds, training_flag)
recon_x_logits, dec_layers = build_decoder(state, prob)
recon_x = tf.nn.sigmoid(recon_x_logits)

# output in logits transform to actual units
pred_y = tf.math.argmax(pred_y_logits, axis=-1)

# get variables for training
enc_vars = []
for layer in enc_layers:
    with tf.variable_scope(layer, reuse=True):
        enc_vars.append(tf.get_variable('kernel'))
        enc_vars.append(tf.get_variable('bias'))
dec_vars = []
for layer in dec_layers:
    with tf.variable_scope(layer, reuse=True):
        dec_vars.append(tf.get_variable('kernel'))
        dec_vars.append(tf.get_variable('bias'))


# compute losses and the network cost
loss_y = digit_loss(pred_y_logits, y)
loss_div = kl_loss(style_state_means, style_state_stds)
loss_recon = recon_loss(recon_x_logits, x)
cost = loss_y + loss_div + 10 * loss_recon
train = tf.train.AdamOptimizer(learn_rate).minimize(cost, var_list=enc_vars+dec_vars)

# initializer for network
init = tf.global_variables_initializer()

#prepare figure for viz
fig = plt.figure(figsize=(20, 15))

with tf.Session() as session:
    # initalize network
    session.run(init)

    # train network
    for iter in range(iterations + 1):
        acc_cost = 0
        acc_cost_y = 0
        acc_cost_div = 0
        acc_cost_recon = 0

        # get randomized indices for this batch
        batch_index = np.random.permutation(dataset_size)

        for b in range(num_batches_per_it):

            # get data for this batch, then upsample and rotate
            this_batch_x = np.zeros([batch_size, img_size, img_size])
            this_batch_theta = np.float32(np.random.normal(size=(batch_size, 1)) * pi / 6)

            for k in range(batch_size):
                this_batch_x[k, :, :] = rmn.generate_image(x_train[batch_index[(b * batch_size) + k], :, :],
                                                           this_batch_theta[k, :], img_size)

            # get desired output
            this_batch_y = y_train[batch_index[(b * batch_size):((b + 1) * batch_size)]]

            _, this_cost, this_loss_y, this_loss_div, this_loss_recon = \
                session.run([train, cost, loss_y, loss_div, loss_recon], feed_dict={
                    x: np.expand_dims(this_batch_x, axis=-1),
                    y: this_batch_y,
                    prob: 0.0,
                    training_flag: 1.0
                })

            acc_cost += this_cost
            acc_cost_y += this_loss_y
            acc_cost_div += this_loss_div
            acc_cost_recon += this_loss_recon

        print("TRAIN " + str(iter) + " Losses: " + str(acc_cost / num_batches_per_it)
              + "\tnumber " + str(acc_cost_y / num_batches_per_it)
              + "\tdivergence " + str(acc_cost_div / num_batches_per_it)
              + "\trecon " + str(acc_cost_recon / num_batches_per_it), flush=True)

        if iter % report_iter == 0:
            acc_cost = 0.0
            acc_cost_y = 0.0
            acc_cost_div = 0.0
            acc_cost_recon = 0.0
            for b in range(num_test_batches):

                # get data for this batch, then upsample and rotate
                this_batch_x = np.zeros([batch_size, img_size, img_size])

                for k in range(batch_size):
                    this_batch_x[k, :, :] = rmn.generate_image(x_test[(b * batch_size) + k, :, :],
                                                               0.0, img_size)

                # get desired output
                this_batch_y = y_test[(b * batch_size):((b + 1) * batch_size)]

                if b  == 0:
                    this_cost, this_loss_y, this_loss_div, this_loss_recon, this_recon_x = \
                        session.run([cost, loss_y, loss_div, loss_recon, recon_x], feed_dict={
                            x: np.expand_dims(this_batch_x, axis=-1),
                            y: this_batch_y,
                            prob: 0.0,
                            training_flag: 0.0
                        })

                    viz_numbers = [3, 2, 1, 18, 4, 8, 11, 0, 61, 9]

                    for viz_i in range(10):
                        ax1 = plt.subplot(3, 8, 2*viz_i+1)
                        ax2 = plt.subplot(3, 8, 2*viz_i+2)
                        ax1.imshow(np.squeeze(this_batch_x[viz_numbers[viz_i], :, :]),vmin=0,vmax=1)
                        ax2.imshow(np.squeeze(this_recon_x[viz_numbers[viz_i], :, :]),vmin=0,vmax=1)
                    plt.show()
                    plt.pause(0.001)

                else:
                    this_cost, this_loss_y, this_loss_div, this_loss_recon = \
                        session.run([cost, loss_y, loss_div, loss_recon], feed_dict={
                            x: np.expand_dims(this_batch_x, axis=-1),
                            y: this_batch_y,
                            prob: 0.0,
                            training_flag: 0.0
                        })

                acc_cost += this_cost
                acc_cost_y += this_loss_y
                acc_cost_div += this_loss_div
                acc_cost_recon += this_loss_recon

            print("TEST " + str(iter) + " Losses: " + str(acc_cost / num_test_batches)
                  + "\tnumber " + str(acc_cost_y / num_test_batches)
                  + "\tdivergence " + str(acc_cost_div / num_test_batches)
                  + "\trecon " + str(acc_cost_recon / num_test_batches), flush=True)
