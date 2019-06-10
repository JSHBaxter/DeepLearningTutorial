import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import time
import rotate_mnist as rmn
import warnings
warnings.filterwarnings('ignore')

pi = 3.141592

# get dataset
mnist = K.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
x_train, x_test = x_train / 255.0, x_test / 255.0

# parameters
pretrain_iterations = 0
iterations = 2000
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

# build network
# inputs
x = tf.placeholder(tf.float32, [None, img_size, img_size,1], name="x")
y = tf.placeholder(tf.int32, [None,], name="y")
training_flag = tf.placeholder_with_default(1.0, shape=(), name="training_flag")
prob_gen = tf.placeholder_with_default(0.0, shape=(), name="prob_gen")

# convolutional network encoder with 3 conv layers, followed by a dense layer
# (maxpooling after first two conv layers)
convE1 = tf.layers.max_pooling2d(
    tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, name="convE1"), 2, 2)
convE2 = tf.layers.max_pooling2d(tf.layers.conv2d(convE1, 128, 5, activation=tf.nn.relu, name="convE2"), 2, 2)
convE3 = tf.layers.conv2d(convE2, 256, 3, activation=tf.nn.relu, name="convE3")
denseE1 = tf.layers.dense(tf.layers.dropout(tf.layers.flatten(convE3), prob_gen), 512, activation=tf.nn.relu, name="denseE1")

# predict output identity and style parameters as internal state
pred_y_logits = tf.layers.dense(tf.layers.dropout(denseE1, prob_gen), 10, name="pred_y_logits")
style_state_means = tf.layers.dense(tf.layers.dropout(denseE1, prob_gen), style_dims,
                                    name="style_state_means", bias_initializer=tf.initializers.zeros())
style_state_stds= tf.layers.dense(tf.layers.dropout(denseE1, prob_gen), style_dims,
                                    name="style_state_stds", bias_initializer=tf.initializers.zeros())

# output in logits transform to actual units
pred_y = tf.math.argmax(pred_y_logits, axis=-1)
y_state = tf.nn.softmax(pred_y_logits, axis=-1)
style_state = style_state_means + \
              training_flag * style_state_stds * tf.random.normal(tf.shape(style_state_stds))

# decoder
denseD1 = tf.layers.dense(tf.concat([y_state, style_state], axis=-1), 512, activation=tf.nn.relu, name='denseD1')
denseD2 = tf.layers.dense(denseD1, int(img_size//4)*int(img_size//4)*32, activation=tf.nn.relu, name='denseD2')
denseD2 = tf.reshape(denseD2,[-1,int(img_size//4),int(img_size//4),32])
convD3 = tf.layers.conv2d_transpose(denseD2, 64, kernel_size=5, strides=[2, 2],
                                    padding="SAME", activation=tf.nn.relu, name='convD3')
convD2 = tf.layers.conv2d_transpose(convD3,32,kernel_size=5, strides=[2, 2],
                                    padding="SAME", activation=tf.nn.relu, name='convD2')
convD1 = tf.layers.conv2d(convD2,1,kernel_size=1, name='convD1')
recon_x = tf.nn.sigmoid(convD1)

# discriminator
switch = tf.placeholder(tf.float32, [None,], name="switch")
prob_disc = tf.placeholder_with_default(0.0, shape=(), name="prob_disc")
switch_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(switch,-1),-1),-1)
disc_input = switch_expanded*x + (1.0-switch_expanded) * recon_x
convM1 = tf.layers.max_pooling2d(
    tf.layers.conv2d(disc_input, 8, 3, activation=tf.nn.relu, name="convM1"), 2, 2)
convM2 = tf.layers.max_pooling2d(tf.layers.conv2d(convM1, 16, 3, activation=tf.nn.relu, name="convM2"), 2, 2)
convM3 = tf.layers.max_pooling2d(tf.layers.conv2d(convM2, 32, 3, activation=tf.nn.relu, name="convM3"), 2, 2)
denseM1 = tf.layers.dense(tf.layers.dropout(tf.layers.flatten(convM3), prob_disc), 128, activation=tf.nn.relu, name="denseM1")
denseM2 = tf.squeeze(tf.layers.dense(tf.layers.dropout(denseM1, prob_disc), 1, name="denseM2"))

# encoder variables
enc_vars = []
for layer in ['convE1','convE3','convE3','denseE1','pred_y_logits','style_state_means','style_state_stds']:
    with tf.variable_scope(layer, reuse=True):
        enc_vars.append(tf.get_variable('kernel'))
        enc_vars.append(tf.get_variable('bias'))

# decoder variables
dec_vars = []
for layer in ['denseD1','denseD2','convD3','convD2','pred_y_logits','convD1']:
    with tf.variable_scope(layer, reuse=True):
        dec_vars.append(tf.get_variable('kernel'))
        dec_vars.append(tf.get_variable('bias'))

# discriminator variables
disc_vars = []
for layer in ['convM1','convM3','convM3','denseM1','denseM2']:
    with tf.variable_scope(layer, reuse=True):
        disc_vars.append(tf.get_variable('kernel'))
        disc_vars.append(tf.get_variable('bias'))

# compute losses and the network cost
loss_y = tf.losses.sparse_softmax_cross_entropy(y, pred_y_logits)
loss_div = tf.math.reduce_mean(tf.math.square(style_state_means) + tf.math.square(style_state_stds)
                               - tf.math.log(tf.math.square(style_state_stds)) - 1)
loss_recon = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=convD1))
loss_disc = tf.losses.sigmoid_cross_entropy(multi_class_labels=switch, logits=denseM2)
loss_conf = (1-tf.math.reduce_max(switch)) * tf.losses.sigmoid_cross_entropy(multi_class_labels=(1-switch), logits=denseM2)
cost_gen = loss_y + loss_div + img_size*loss_recon + loss_conf
cost_disc = loss_disc
train_gen = tf.train.AdamOptimizer(learn_rate).minimize(cost_gen, var_list=enc_vars+dec_vars)
train_disc = tf.train.AdamOptimizer(0.01*learn_rate).minimize(cost_disc, var_list=disc_vars)

# initializer for network
init = tf.global_variables_initializer()

#prepare figure for viz
fig = plt.figure(figsize=(20, 15))
plt.ion()
plt.show()


def iteration_train_disc(iter):
    acc_cost_disc = 0

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
        this_batch_y = y_train[(b * batch_size):((b + 1) * batch_size)]

        _, this_cost_disc = session.run([train_disc, loss_disc], feed_dict={
            x: np.expand_dims(this_batch_x, axis=-1),
            y: this_batch_y,
            prob_disc: 0.5,
            prob_gen: 0.0,
            training_flag: 1.0,
            switch: (np.random.uniform(size=[batch_size,]) > 0.5).astype(float)
        })
        acc_cost_disc += this_cost_disc

    print("TRAIN DISC " + str(iter) + " number " + str(acc_cost_disc / num_batches_per_it), flush=True)


def iteration_train_gen(iter, use_disc):
    acc_cost = 0
    acc_cost_y = 0
    acc_cost_div = 0
    acc_cost_recon = 0

    # get randomized indices for this batch
    batch_index = np.random.permutation(dataset_size)
    if use_disc:
        this_switch = np.zeros([batch_size,])
    else:
        this_switch = np.ones([batch_size,])

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
            session.run([train_gen, cost_gen, loss_y, loss_div, loss_recon], feed_dict={
                x: np.expand_dims(this_batch_x, axis=-1),
                y: this_batch_y,
                prob_gen: 0.05,
                prob_disc: 0.0,
                training_flag: 1.0,
                switch: this_switch
            })

        acc_cost += this_cost
        acc_cost_y += this_loss_y
        acc_cost_div += this_loss_div
        acc_cost_recon += this_loss_recon

    print("TRAIN GEN " + str(iter) + " Losses: " + str(acc_cost / num_batches_per_it)
          + "\tnumber " + str(acc_cost_y / num_batches_per_it)
          + "\tdivergence " + str(acc_cost_div / num_batches_per_it)
          + "\trecon " + str(acc_cost_recon / num_batches_per_it), flush=True)


def iteration_test_gen(iter):
    acc_cost_y = 0
    acc_cost_div = 0
    acc_cost_recon = 0

    for b in range(num_test_batches):

        # get data for this batch, then upsample and rotate
        this_batch_x = np.zeros([batch_size, img_size, img_size])

        for k in range(batch_size):
            this_batch_x[k, :, :] = rmn.generate_image(x_test[(b * batch_size) + k, :, :],
                                                       0.0, img_size)

        # get desired output
        this_batch_y = y_test[(b * batch_size):((b + 1) * batch_size)]

        if b % viz_iter == viz_iter - 1:
            this_loss_y, this_loss_div, this_loss_recon, this_recon_x = \
                session.run([loss_y, loss_div, loss_recon, recon_x], feed_dict={
                    x: np.expand_dims(this_batch_x, axis=-1),
                    y: this_batch_y,
                    prob_gen: 0.0,
                    prob_disc: 0.0,
                    training_flag: 0.0,
                    switch: np.zeros([batch_size,])
                })

            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            ax1.imshow(np.squeeze(this_batch_x[0, :, :]), vmin=0, vmax=1)
            ax2.imshow(np.squeeze(this_recon_x[0, :, :]), vmin=0, vmax=1)
            fig.savefig('VAA_Iter' + str(iter) + 'Batch' + str(b + 1) + '.pdf')
            # plt.pause(0.001)

        else:
            this_loss_y, this_loss_div, this_loss_recon = \
                session.run([loss_y, loss_div, loss_recon], feed_dict={
                    x: np.expand_dims(this_batch_x, axis=-1),
                    y: this_batch_y,
                    prob_gen: 0.0,
                    prob_disc: 0.0,
                    training_flag: 0.0,
                    switch: np.zeros([batch_size,])
                })

        acc_cost_y += this_loss_y
        acc_cost_div += this_loss_div
        acc_cost_recon += this_loss_recon

    print("TEST GEN " + str(iter) + " Losses: number " + str(acc_cost_y / num_test_batches)
          + "\tdivergence " + str(acc_cost_div / num_test_batches)
          + "\trecon " + str(acc_cost_recon / num_test_batches), flush=True)


with tf.Session() as session:
    # initalize network
    session.run(init)

    iter_disc = 0
    iter_gen = 0

    for iter in range(pretrain_iterations):
        iteration_train_disc(iter_disc)
        iter_disc += 1

    while iter_gen <= iterations:
        for i in range(report_iter):
            iteration_train_gen(iter_gen, iter_disc > 0)
            iter_gen += 1
        for i in range(report_iter):
            iteration_train_disc(iter_disc)
            iter_disc += 1
        iteration_test_gen(iter_gen)
