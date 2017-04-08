
import tensorflow as tf
import itertools as it
import numpy as np

import reader

class ModelConfig(object):
    num_units = 10
    num_classes = 100

    epochs = 200

    num_seq = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    test_step_size = 3

# test1 = [i for i in range(0, num_classes, 1)]
# test2 = [i for i in range(0, num_classes, 2)]
#
#
# inputs = [test1, test2]

def get_input(config):
    inputs = [[j for j in range(0, config.num_classes, i)] for i in config.num_seq]

    # tf_inputs = [tf.convert_to_tensor([*[i for i in input]], dtype=tf.int32) for input in it.zip_longest(*inputs, fillvalue=0)]
    # targets = [tf.convert_to_tensor(inputs, dtype=tf.int32) for inputs in it.zip_longest(*inputs, fillvalue=0)]

    tf_inputs = tf.convert_to_tensor(list(it.zip_longest(*inputs, fillvalue=0)), dtype=tf.int32)
    targets = tf.convert_to_tensor(list(it.zip_longest(*inputs, fillvalue=0)), dtype=tf.int32)
    sequence_lengths = tf.convert_to_tensor([len(i) for i in inputs], dtype=tf.int32)

    return tf_inputs, targets, sequence_lengths


def get_test_input(config):
    test_input = tf.convert_to_tensor([[i] for i in range(0, config.num_classes, config.test_step_size)], dtype=tf.int32)
    test_targets = tf.convert_to_tensor([[i] for i in range(0, config.num_classes, config.test_step_size)], dtype=tf.int32)

    return test_input, test_targets


def sequence_cross_entropy(labels, logits, sequence_lengths):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if sequence_lengths is not None:
        loss_sum = tf.reduce_sum(cross_entropy, axis=0)
        return tf.truediv(loss_sum, tf.cast(sequence_lengths, tf.float32))
    else:
        return tf.reduce_mean(cross_entropy, axis=0)


class Model:

    def __init__(self, config,  inputs, targets, batch_size, sequence_lengths=None):
        global_step = tf.Variable(0, trainable=False)

        # cell = tf.contrib.rnn.BasicLSTMCell(num_units=size, forget_bias=0.0, state_is_tuple=True)
        # cell = tf.contrib.rnn.LSTMBlockCell(num_units=size, forget_bias=1.0)
        #
        # with tf.name_scope("initial_state"):
        #     initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        #     # tf.summary.tensor_summary('initial_state', initial_state)

        embedding = tf.get_variable("embedding", [config.num_classes, config.num_units], dtype=tf.float32)
        _inputs = tf.nn.embedding_lookup(embedding, inputs)

        # outputs, state = tf.contrib.rnn.static_rnn(cell, _inputs, initial_state=initial_state, scope="rnn", sequence_length=sequence_lengths)

        lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=config.num_units, forget_bias=1.0, cell_clip=None, use_peephole=False)

        initial_state = (tf.zeros([batch_size, config.num_units], tf.float32), tf.zeros([batch_size, config.num_units], tf.float32))

        output, state = lstm(_inputs, initial_state=initial_state, dtype=None, sequence_length=sequence_lengths, scope="rnn")



        softmax_w = tf.get_variable("softmax_w", [config.num_units, config.num_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.num_classes], dtype=tf.float32)
        # logits = [tf.matmul(output, softmax_w) + softmax_b for output in outputs]

        _output = tf.reshape(output, [-1, config.num_units])
        _logits = tf.matmul(_output, softmax_w) + softmax_b

        logits = tf.reshape(_logits, [-1, batch_size, config.num_classes])

        loss = tf.reduce_mean(sequence_cross_entropy(labels=targets, logits=logits, sequence_lengths=sequence_lengths))

        starting_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 10, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        trainable_vars = tf.trainable_variables()

        train_step = optimizer.minimize(loss, var_list=trainable_vars, global_step=global_step)

        self.learning_rate = learning_rate
        self.logits = logits
        self.loss = loss
        self.train_step = train_step


def train(config):
    with tf.Session() as session:

        summary_writer = tf.summary.FileWriter("logs/")

        with tf.name_scope("Train"):
            inputs, targets, sequence_lengths = get_input(config)

            with tf.variable_scope("Model", reuse=None):
                train_model = Model(config, inputs, targets, len(config.num_seq), sequence_lengths=sequence_lengths)

            tf.summary.scalar("loss", train_model.loss)
            tf.summary.scalar("learning rate", train_model.learning_rate)



        merged = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        summary_writer.add_graph(session.graph)

        session.run(init_op)

        for i in range(config.epochs):
            if i % 10 == 0:
                print(i)
                summary = session.run(merged)
                summary_writer.add_summary(summary, i)

            session.run(train_model.train_step)



def test(config):
    with tf.Session() as session:

        summary_writer = tf.summary.FileWriter("logs/")

        with tf.name_scope("Test"):
            test_inputs, test_targets = get_test_input(config)

            with tf.variable_scope("Model", reuse=True):
                test_model = Model(config, test_inputs, test_targets, 1)

            tf.summary.scalar("loss", test_model.loss)

        merged = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        session.run(init_op)


        logits = test_model.logits
        loss = test_model.loss

        summary, out, targ, loss = session.run([merged, logits, test_targets, loss])

        # np.set_printoptions(precision=5)
        pretty_out= np.array([[np.argmax(batch) for batch in batches] for batches in out])
        pretty_targets = np.array([i.tolist() for i in targ])

        print(loss)
        print(np.swapaxes(np.array([pretty_out, pretty_targets]), 0, 2))


if __name__ == '__main__':
    config = ModelConfig()
    train(config)

    # dataset = reader.dataset160()
    # print(dataset.partition(10)[0])

