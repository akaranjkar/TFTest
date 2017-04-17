import tensorflow as tf

def summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(var - mean))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

# placeholders are inputs to the model
# variables are things that get trained. like weights and biases
# declare variables and linear model
with tf.name_scope('weight'):
    W = tf.Variable([.3],tf.float32)
    summaries(W)
with tf.name_scope('bias'):
    b = tf.Variable([-.3],tf.float32)
    summaries(b)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# initialize variables
# init = tf.global_variables_initializer()
# sess = tf.Session()

# loss
y = tf.placeholder(tf.float32) # desired output
squared_deltas = tf.square(linear_model - y) # square differences between output and desired

with tf.name_scope('loss'):
    loss = tf.reduce_sum(squared_deltas)
    summaries(loss)

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