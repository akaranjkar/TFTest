import tensorflow as tf

# placeholders are inputs to the model
# variables are things that get trained. like weights and biases
# declare variables and linear model
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# initialize variables
init = tf.global_variables_initializer()
sess = tf.Session()

# loss
y = tf.placeholder(tf.float32) # desired output
squared_deltas = tf.square(linear_model - y) # square differences between output and desired
loss = tf.reduce_sum(squared_deltas)

# optimizer
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b]))