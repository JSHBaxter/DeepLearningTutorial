import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


# Generate image
def generate_image(size, snr, radius1, radius2, rotate, flip):
    img = np.zeros((size,size), dtype='float32')
    idx_x, idx_y = np.meshgrid(range(size), range(size))
    idx_x, idx_y = idx_x / size, idx_y / size
    img[idx_x >= idx_y] = -1.0
    img[idx_x < idx_y] = 1.0
    img[(idx_x-0.75)**2 + (idx_y-0.25)**2 < radius1**2] = 0.0
    img[(idx_x-0.25)**2 + (idx_y-0.75)**2 < radius2**2] = 0.0
    if rotate:
        img = np.transpose(img)
    if flip:
        img = np.flip(img)
    return img + np.random.normal(size=img.shape) / snr


def generate_mask(size,radius1,radius2, rotate, flip):
    img = np.zeros((size,size), dtype='int32')
    idx_x, idx_y = np.meshgrid(range(size), range(size))
    idx_x, idx_y = idx_x / size, idx_y / size
    img[idx_x >= idx_y] = 0
    img[idx_x < idx_y] = 1
    img[(idx_x-0.75)**2 + (idx_y-0.25)**2 < radius1**2] = 2
    img[(idx_x-0.25)**2 + (idx_y-0.75)**2 < radius2**2] = 3
    if rotate:
        img = np.transpose(img)
    if flip:
        img = np.flip(img)
    return img


# Build database from images
def build_database(number,size,snr):
    database_i = np.zeros((number,size,size,1), dtype='float32')
    database_o = np.zeros((number,size,size), dtype='int32')
    for n in range(number):
        radius1, radius2 = 0.1*np.random.uniform()+0.2, 0.1*np.random.uniform()+0.2
        rotate, flip = np.random.uniform() > 0.5, np.random.uniform() > 0.5
        database_i[n,:,:,0] = generate_image(size, snr, radius1, radius2, rotate, flip)
        database_o[n,:,:] = generate_mask(size, radius1, radius2, rotate, flip)
    return database_i, database_o


# Single iteration of a DenseCRF mean field approximation iteration:
#
# Using a fairly dense model with Gaussian weighted edges (with respect to distance). That way, the message passing
# can be expressed as the same convolution filter (weighted) across all image channels followed by the point-wise
# compatibility transform. (Two can be combined as a single separable conv2d operation.)
#
# We are performing softmax first, that way, q_in/q_out is expressed as logits, same as p_in
def DenseCRF_MeanApproxIteration(q_in, p_in, kernel, compatibility):
    # perform softmax on Q to put in [0,1] range
    q_in = tf.nn.softmax(q_in)

    # perform message passing followed by the compatibility transform
    q_weighted = tf.nn.separable_conv2d(q_in, kernel, compatibility, strides=[1,1,1,1], padding='SAME')

    # add in unary potentials and output
    q_out = p_in - q_weighted
    return q_out


# Create Gaussian kernel
def get_gaussian_kernel(size, std):
    d = tf.distributions.Normal(0.0, std)
    values = d.prob(tf.range(start=-size, limit=size+1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', values, values)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    return gauss_kernel




# Parameters for training
learn_rate = 0.01
iterations = 1000
crf_iterations = 30
img_size = 32
img_snr = 1
training_size = 25
testing_size = 25

# Get training set
training_i, training_o = build_database(training_size, img_size, img_snr)

# Build network
# Inputs
x = tf.placeholder(tf.float32, [None, img_size, img_size, 1], name="x")
y = tf.placeholder(tf.int32, [None, img_size, img_size], name="y")

# Network
conv1 = tf.layers.conv2d(x, 32, 3, padding='SAME', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(conv1, 128, 1, padding='SAME', activation=tf.nn.relu)
conv3 = tf.layers.conv2d(conv2, 4, 1, padding='SAME')

# Weights for CNN
k_size = 10
k = tf.broadcast_to(get_gaussian_kernel(k_size,5.0), [2*k_size+1, 2*k_size+1, 4, 1])
compatibility = tf.get_variable("compatibility", [1,1,4,4], dtype=tf.float32, initializer=tf.initializers.zeros())

# Iterations for CNN
Q = conv3
for i in range(crf_iterations):
    Q = DenseCRF_MeanApproxIteration(Q, conv3, k, compatibility)

#outputs
pred_logits = Q
pred_y = tf.nn.softmax(pred_logits)
out_y = tf.math.argmax(pred_logits,axis=-1)
acc = 1.0 - tf.math.reduce_mean(tf.math.abs(tf.one_hot(y, 4)-pred_y))

#confusion matrix
confusion = tf.math.confusion_matrix(tf.reshape(y, [-1]), tf.reshape(out_y, [-1]), 4)

# Costs and initializers
cost =  tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits))
train = tf.train.AdamOptimizer(learn_rate).minimize(cost)
init = tf.global_variables_initializer()


# Initializer for network
init = tf.global_variables_initializer()








with tf.Session() as session:

    #initalize network
    session.run(init)

    #train network
    for iter in range(iterations+1):

        _, this_cost, this_acc, this_compatibility, this_confusion, this_y = \
            session.run([train, cost, acc, compatibility, confusion, out_y], feed_dict={
                x: training_i,
                y: training_o
            })

        log_iter = 100
        if(iter > 0 and iter % log_iter != 1):
            backstep = 9
            if crf_iterations == 0:
                backstep -= 4
            for j in range(backstep):
                print("\033[F" + "\033[K", end='')
        if( iter % log_iter == 0):
            plt.imshow(this_y[0,:,:])
            plt.show()
            time.sleep(0.001)

        print(str(iter) + " Losses: " + str(this_cost) + " " + str(this_acc))
        if crf_iterations > 0:
            print(str(this_compatibility))
        print(str(this_confusion))




