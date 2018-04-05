import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# for tensor board use


# parameters
learning_rate = 0.1
num_steps = 5000
batch_size = 128
display_step = 500

dropout_rate = 0.1
keep_prob_rate = 1-dropout_rate
# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 128
n_hidden_3 = 64
num_input = 784
num_classes = 10

# load mnist dataset
# use tensorflow api
mnist = input_data.read_data_sets('./tf/data', one_hot=True)

# input and output for model
# for tf.placeholder
# (dtype, shape, name)
# None is same as -1 in pytorch
X = tf.placeholder('float', [None, num_input], 'input')
Y = tf.placeholder('float', [None, num_classes], 'out') # as we use one-hot encoding

# define the model
# for the linear y = wx+b , I use the api tf.layers.dense(), can be accomplished by tf.matmul, t.add as well

weights = {
    'h1':tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out':tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':tf.Variable(tf.random_normal([n_hidden_3])),
    'out':tf.Variable(tf.random_normal([num_classes]))
}

# model net
def model_net(x):
    # layer1
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer1 = tf.nn.relu(layer1)
    #layer1 = tf.nn.dropout(layer1, keep_prob=0.9)
    # layer 2
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    #layer2 = tf.nn.relu(layer2)
    #layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob_rate)
    # layer 3
    layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
    #layer3 = tf.nn.relu(layer3)
    #layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob_rate)

    # output layer
    output = tf.add(tf.matmul(layer3, weights['out']) , biases['out'])
    # use soft max
    #output = tf.nn.softmax(output)
    return output

# model construction
logits = model_net(X)
prediction = tf.nn.softmax(logits)

# tf.train.exponential_decay
# 学习率的自动衰减
# decay_rate是衰减系数， decay_steps是衰减速度，learning_rate是初始学习率
current_epoch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                           global_step=current_epoch,
                                           decay_steps=num_steps,
                                           decay_rate=0.03)
#
tf.summary.scalar('lr', learning_rate)

# Define the loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, global_step=current_epoch)

#
tf.summary.scalar('loss', loss_op)

# evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#
tf.summary.scalar('acc', accuracy)

# start running

# init the variables
init = tf.global_variables_initializer()

# tf summary
#tf.summary.image()

with tf.Session() as sess:


    # step 50 create a log
    train_writer = tf.summary.FileWriter('./tf/log', sess.graph)
    merged = tf.summary.merge_all()

    sess.run(init)

    for step in range(1, num_steps+1):
        # update current
        current_epoch = step
        # get dataset batch
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # run optmization option
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
        if step % display_step == 0 or step == 1:
            # loss and acc
            loss, acc, summary = sess.run([loss_op, accuracy, merged], feed_dict={X:batch_x, Y:batch_y})
            # tensorboard summary
            train_writer.add_summary(summary, step)
            # print log information
            print('Step {} - Loss: {:.4f} ; Accuracy: {:.4f} ; Learning rate : {}'.format(step, loss, acc, sess.run(learning_rate)))
    print("Training Finished")
            # update tensor board

    # Test
    print("Test accuracy: {:.4f}".format(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})))