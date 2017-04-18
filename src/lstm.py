import tensorflow as tf
import numpy as np

def sequence_cross_entropy(labels, logits, sequence_lengths):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if sequence_lengths is not None:
        loss_sum = tf.reduce_sum(cross_entropy, axis=0)
        return tf.truediv(loss_sum, tf.cast(sequence_lengths, tf.float32))
    else:
        return tf.reduce_mean(cross_entropy, axis=0)


class Model:

    def __init__(self, config,  inputs, targets, batch_size, sequence_lengths=None, is_training=False):
        global_step = tf.Variable(0, trainable=False)

        embedding = tf.get_variable("embedding", [config.num_classes, config.num_units], dtype=tf.float32)
        _inputs = tf.nn.embedding_lookup(embedding, inputs)

        lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=config.num_units, forget_bias=0, cell_clip=None, use_peephole=False)

        initial_state = (tf.zeros([batch_size, config.num_units], tf.float32), tf.zeros([batch_size, config.num_units], tf.float32))

        output, state = lstm(_inputs, initial_state=initial_state, dtype=None, sequence_length=sequence_lengths, scope="rnn")

        softmax_w = tf.get_variable("softmax_w", [config.num_units, config.num_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.num_classes], dtype=tf.float32)

        _output = tf.reshape(output, [-1, config.num_units])
        _logits = tf.matmul(_output, softmax_w) + softmax_b

        logits = tf.reshape(_logits, [-1, batch_size, config.num_classes])

        loss = tf.reduce_mean(sequence_cross_entropy(labels=targets, logits=logits, sequence_lengths=sequence_lengths))

        self.logits = logits
        self.loss = loss

        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(trainable_vars)

        if not is_training:
            return

        starting_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 10, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        train_step = optimizer.minimize(loss, var_list=trainable_vars, global_step=global_step)

        self.learning_rate = learning_rate
        self.train_step = train_step


def train(config, summary_writer):
    batch_size = config.batch_size

    with tf.name_scope("Train"):

        inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="inputs")
        targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
        sequence_lengths = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lengths")

        with tf.variable_scope("Model", reuse=None):
            train_model = Model(config, inputs, targets, batch_size, sequence_lengths=sequence_lengths, is_training=True)

        tf.summary.scalar("loss", train_model.loss)
        tf.summary.scalar("learning rate", train_model.learning_rate)



    merged = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:

        session.run(init_op)

        train_dataset = config.train_dataset

        for i in range(config.epochs):

            train_dataset.shuffle()
            batches = train_dataset.partition(batch_size)
            # print("# of batches: %i" % len(batches))

            for j, (_inputs, _targets, _sequence_lengths) in enumerate(batches):

                feed_dict = {inputs: _inputs,
                             targets: _targets,
                             sequence_lengths: _sequence_lengths}

                if j == 0:
                    print(i)
                    summary = session.run(merged, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, i)

                session.run(train_model.train_step, feed_dict=feed_dict)

        train_model.saver.save(session, "checkpoints/model.ckpt")


def test(config, summary_writer):
    batch_size = 1

    with tf.name_scope("Test"):
        inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="inputs")
        targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
        sequence_lengths = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lengths")

        with tf.variable_scope("Model", reuse=True):
            test_model = Model(config, inputs, targets, batch_size, sequence_lengths=sequence_lengths)

        tf.summary.scalar("loss", test_model.loss)

    with tf.Session() as session:
        test_model.saver.restore(session, "checkpoints/model.ckpt")

        test_dataset = config.test_dataset
        test_dataset.shuffle()
        batches = test_dataset.partition(batch_size)

        for _inputs, _targets, _sequence_lengths in batches:
            feed_dict = {inputs: _inputs,
                         targets: _targets,
                         sequence_lengths: _sequence_lengths}

            logits = test_model.logits
            loss = test_model.loss

            # summary, out, targ, loss = session.run([merged, logits, test_targets, loss])
            out, loss = session.run([logits, loss], feed_dict=feed_dict)

            # np.set_printoptions(precision=5)
            pretty_out= np.array([[np.argmax(batch) for batch in _batches] for _batches in out])
            pretty_targets = np.array([i.tolist() for i in _targets])

            print(loss)
            print(np.swapaxes(np.array([pretty_out, pretty_targets]), 0, 2))
