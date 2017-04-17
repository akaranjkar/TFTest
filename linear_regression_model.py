import tensorflow as tf

# placeholders are inputs to the model
# variables are things that get trained. like weights and biases
# declare variables and linear model
with tf.name_scope('variables'):
    with tf.name_scope('weight'):
        W = tf.Variable([.3],tf.float32)
    with tf.name_scope('bias'):
        b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# initialize variables
# init = tf.global_variables_initializer()
# sess = tf.Session()

# loss
y = tf.placeholder(tf.float32) # desired output
squared_deltas = tf.square(linear_model - y) # square differences between output and desired
loss = tf.reduce_sum(squared_deltas)

# optimizer
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/', sess.graph)

    # sess.run(init)
    for i in range(1000):
        summary,_ = sess.run([merged, train], {x:[1,2,3,4], y:[0,-1,-2,-3]})
        train_writer.add_summary(summary,i)

    # print(sess.run([W,b]))
    train_writer.close()