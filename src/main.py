
import tensorflow as tf
import itertools as it
import numpy as np

import reader

size = 10
num_classes = 100

num_seq = [1, 2, 4, 5, 6, 7, 8, 9, 10]
test_step_size = 3

# test1 = [i for i in range(0, num_classes, 1)]
# test2 = [i for i in range(0, num_classes, 2)]
#
#
# inputs = [test1, test2]

def get_input():
    inputs = [[j for j in range(0, num_classes, i)] for i in num_seq]

    tf_inputs = [tf.convert_to_tensor([*[i for i in input]], dtype=tf.int32) for input in it.zip_longest(*inputs, fillvalue=0)]
    targets = [tf.convert_to_tensor(inputs, dtype=tf.int32) for inputs in it.zip_longest(*inputs, fillvalue=0)]
    sequence_lengths = [len(i) for i in inputs]

    return tf_inputs, targets, sequence_lengths


def get_test_input():
    test_input = [tf.convert_to_tensor([i], dtype=tf.int32) for i in range(0, num_classes, test_step_size)]
    test_targets = [tf.convert_to_tensor([i], dtype=tf.int32) for i in range(0, num_classes, test_step_size)]

    return test_input, test_targets


def sequence_cross_entropy(labels, logits, sequence_lengths):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if sequence_lengths is not None:
        loss_sum = tf.reduce_sum(cross_entropy, axis=0)
        return tf.truediv(loss_sum, tf.convert_to_tensor(sequence_lengths, dtype=tf.float32))
    else:
        return tf.reduce_mean(cross_entropy, axis=0)


class model:

    def __init__(self, inputs, targets, batch_size, sequence_lengths=None):
        global_step = tf.Variable(0, trainable=False)
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=size, forget_bias=0.0, state_is_tuple=True)

        with tf.name_scope("initial_state"):
            initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            # tf.summary.tensor_summary('initial_state', initial_state)

        embedding = tf.get_variable("embedding", [num_classes, size], dtype=tf.float32)
        _inputs = [tf.nn.embedding_lookup(embedding, input) for input in inputs]


        outputs, state = tf.contrib.rnn.static_rnn(cell, _inputs, initial_state=initial_state, scope="rnn", sequence_length=sequence_lengths)

        softmax_w = tf.get_variable("softmax_w", [size, num_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
        logits = [tf.matmul(output, softmax_w) + softmax_b for output in outputs]

        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.96, staircase=True)

        loss = tf.reduce_mean(sequence_cross_entropy(labels=targets, logits=logits, sequence_lengths=sequence_lengths))

        optimizer = tf.train.AdamOptimizer(learning_rate)

        trainable_vars = tf.trainable_variables()

        train_step = optimizer.minimize(loss, var_list=trainable_vars, global_step=global_step)

        self.learning_rate = learning_rate
        self.logits = logits
        self.loss = loss
        self.train_step = train_step


def train():
    with tf.Session() as session:

        summary_writer = tf.summary.FileWriter("logs/")

        with tf.name_scope("Train"):
            inputs, targets, sequence_lengths = get_input()

            with tf.variable_scope("Model", reuse=None):
                train_model = model(inputs, targets, len(num_seq), sequence_lengths)

            tf.summary.scalar("loss", train_model.loss)
            tf.summary.scalar("learning rate", train_model.learning_rate)

        with tf.name_scope("Test"):
            test_inputs, test_targets = get_test_input()

            with tf.variable_scope("Model", reuse=True):
                test_model = model(test_inputs, test_targets, 1)

            tf.summary.scalar("loss", test_model.loss)

        merged = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        summary_writer.add_graph(session.graph)

        session.run(init_op)

        for i in range(50):
            if i % 10 == 0:
                print(i)
                summary = session.run(merged)
                summary_writer.add_summary(summary, i)

            session.run(train_model.train_step)


        logits = test_model.logits
        loss = test_model.loss

        summary, out, targ, loss = session.run([merged, logits, test_targets, loss])

        # np.set_printoptions(precision=5)
        pretty_out= np.array([[np.argmax(batch) for batch in batches] for batches in out])
        pretty_targets = np.array([i.tolist() for i in targ])

        print(loss)
        print(np.swapaxes(np.array([pretty_out, pretty_targets]), 0, 2))



if __name__ == '__main__':
    # train()

    dataset = reader.dataset160()


    print(dataset.partition(10)[0])

