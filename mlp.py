from tensorflow.examples.tutorials.mnist import input_data
import cProfile

mnist_data = input_data.read_data_sets("/tmp/data", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
epochs = 15
batch_size = 100
display_step = 1

# Network shape
# Image size 28*28
input_size = 784
h1_size = 256
h2_size = 256
# Output 10 classes
output_size = 10

# Input and output
x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

# Weights
h1_weights = tf.Variable(tf.random_normal([input_size, h1_size]))
h2_weights = tf.Variable(tf.random_normal([h1_size, h2_size]))
output_weights = tf.Variable(tf.random_normal([h2_size, output_size]))

# Biases
h1_bias = tf.Variable(tf.random_normal([h1_size]))
h2_bias = tf.Variable(tf.random_normal([h2_size]))
output_bias = tf.Variable(tf.random_normal([output_size]))


# Feed forward
def forward(input_data):
    h1_output = tf.add(tf.matmul(input_data, h1_weights), h1_bias)
    h1_output = tf.nn.relu(h1_output)

    h2_output = tf.add(tf.matmul(h1_output, h2_weights), h2_bias)
    h2_output = tf.nn.relu(h2_output)

    output_output = tf.add(tf.matmul(h2_output, output_weights), output_bias)
    return output_output


pred = forward(x)

# Define cost function
# Softmax activation on output layer and calculate cross entropy. Then reduce mean
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# Define optimizer. Adam/Adagrad/GradientDescent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    pr = cProfile.Profile()
    pr.enable()

    for epoch in range(epochs):
        avg_cost = 0
        total_batches = int(mnist_data.train.num_examples / batch_size)
        for batch in range(total_batches):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost_function], {x: batch_x, y: batch_y})
            avg_cost += c / batch_size

        if epoch % display_step == 0:
            print("Iteration: ", epoch, " Cost: ", avg_cost)

    # Run on test data
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print(sess.run(accuracy, {x: mnist_data.test.images, y: mnist_data.test.labels}) * 100)
    # print("Accuracy: ", accuracy.eval({x: mnist_data.test.images, y: mnist_data.test.labels}))
    pr.disable()
    pr.print_stats()
