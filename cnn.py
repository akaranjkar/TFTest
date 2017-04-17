from tensorflow.examples.tutorials.mnist import input_data
import cProfile

mnist_data = input_data.read_data_sets("/tmp/data", one_hot=True)

import tensorflow as tf

learning_rate = 0.001
epochs = 200000
batch_size = 128
display_step = 1

# Network shape
input_size = 784
output_size = 10
# Probability to keep units
dropout = 0.75

# Input and output
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])
keep = tf.placeholder(tf.float32)

# Weights
# 5*5 filter, 1 input, 32 outputs
conv1_weights = tf.Variable(tf.random_normal([5, 5, 1, 32]))
# 5*5 filter, 32 input, 64 outputs
conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 64]))
# 7*7*64 inputs, 1024 outputs
fc_weights = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
# Fully connected layer
output_weights = tf.Variable(tf.random_normal([1024, output_size]))

# Biases
conv1_bias = tf.Variable(tf.random_normal([32]))
conv2_bias = tf.Variable(tf.random_normal([64]))
fc_bias = tf.Variable(tf.random_normal([1024]))
output_bias = tf.Variable(tf.random_normal([output_size]))


# Convolutional network
def convolutional_network(input_data):
    # Reshape the input picture
    input_data = tf.reshape(input_data, [-1, 28, 28, 1])

    # Convolutional layer 1
    # Convolution
    conv1_output = tf.nn.conv2d(input_data, conv1_weights, [1, 1, 1, 1], 'SAME')
    conv1_output = tf.nn.bias_add(conv1_output, conv1_bias)
    conv1_output = tf.nn.relu(conv1_output)
    # Pooling
    conv1_output = tf.nn.max_pool(conv1_output, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # Convolutional layer 2
    conv2_output = tf.nn.conv2d(conv1_output, conv2_weights, [1, 1, 1, 1], 'SAME')
    conv2_output = tf.nn.bias_add(conv2_output, conv2_bias)
    conv2_output = tf.nn.relu(conv2_output)
    # Pooling
    conv2_output = tf.nn.max_pool(conv2_output, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # Fully connected layer
    fc1_output = tf.reshape(conv2_output, [-1, fc_weights.get_shape().as_list()[0]])
    fc1_output = tf.add(tf.matmul(fc1_output, fc_weights), fc_bias)
    fc1_output = tf.nn.relu(fc1_output)
    fc1_output = tf.nn.dropout(fc1_output, keep)

    # Output layer
    output_output = tf.add(tf.matmul(fc1_output, output_weights), output_bias)
    return output_output


pred = convolutional_network(x)

# Define cost function
# Softmax activation on output layer and calculate cross entropy. Then reduce mean
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

# Model evaluation
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    pr = cProfile.Profile()
    pr.enable()

    step = 1
    while step * batch_size < epochs:
        batch_x, batch_y = mnist_data.train.next_batch(batch_size)
        sess.run(optimizer, {x: batch_x, y: batch_y, keep: dropout})

        if step % display_step == 0:
            loss, acc = sess.run([cost_function, accuracy], {x: batch_x, y: batch_y, keep: dropout})
            print("Iteration: ", step * batch_size, " Loss: ", loss, " Accuracy: ", acc * 100)
        step += 1
    # Test accuracy
    print(sess.run(accuracy, {x: mnist_data.test.images[:256], y: mnist_data.test.labels[:256], keep: 1}))
    # print("Accuracy: ", accuracy.eval({x: mnist_data.test.images, y: mnist_data.test.labels}))
    pr.disable()
    pr.print_stats()