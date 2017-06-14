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
    def __init__(self, config, inputs, targets, batch_size, sequence_lengths=None, is_training=False):
        global_step = tf.Variable(0, trainable=False)

        embedding = tf.get_variable("embedding", [config.num_input_classes, config.num_units], dtype=tf.float32)
        _inputs = tf.nn.embedding_lookup(embedding, inputs)

        fw_lstm = tf.contrib.rnn.LSTMBlockFusedCell(num_units=config.num_units, forget_bias=0, cell_clip=None,
                                                    use_peephole=True)
        # bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)

        initial_state = (
            tf.zeros([batch_size, config.num_units], tf.float32), tf.zeros([batch_size, config.num_units], tf.float32))

        fw_output, fw_state = fw_lstm(_inputs, initial_state=initial_state, dtype=None,
                                      sequence_length=sequence_lengths, scope="fw_rnn")

        keep_prop = tf.Variable(1, trainable=False, dtype=tf.float32)
        if is_training:
            fw_output = tf.nn.dropout(fw_output, keep_prop)

        # bw_output, bw_state = bw_lstm(fw_output, initial_state=initial_state, dtype=None,
        #                               sequence_length=sequence_lengths, scope="bw_rnn")
        #
        # output = bw_output
        output = fw_output
        # if is_training:
        #     output = tf.nn.dropout(output, keep_prop)

        softmax_w = tf.get_variable("softmax_w", [config.num_units, config.num_output_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.num_output_classes], dtype=tf.float32)

        _output = tf.reshape(output, [-1, config.num_units])
        _logits = tf.matmul(_output, softmax_w) + softmax_b

        logits = tf.reshape(_logits, [-1, batch_size, config.num_output_classes])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            fw_weights = tf.get_variable("fw_rnn/weights")
            bw_weights = tf.get_variable("bw_rnn/weights")

        l2_loss_vars = [embedding, fw_weights, bw_weights, softmax_w]

        l2_loss = 0

        for var in l2_loss_vars:
            l2_loss += tf.nn.l2_loss(var)

        cross_entropy_loss = tf.reduce_mean(sequence_cross_entropy(labels=targets,
                                                                   logits=logits,
                                                                   sequence_lengths=sequence_lengths))

        self.logits = logits
        self.cross_entropy_loss = cross_entropy_loss
        self.keep_prop = keep_prop

        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(trainable_vars)

        if not is_training:
            return

        learning_rate = tf.train.exponential_decay(config.starting_learning_rate, global_step, config.decay_steps,
                                                   config.decay_rate, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        loss = cross_entropy_loss + l2_loss
        train_step = optimizer.minimize(loss, var_list=trainable_vars, global_step=global_step)

        self.learning_rate = learning_rate
        self.train_step = train_step


def train(config, summary_writer, should_print=False, should_validate=False):
    batch_size = config.batch_size

    with tf.name_scope("Train"):

        inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="inputs")
        targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
        sequence_lengths = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lengths")

        with tf.variable_scope("Model", reuse=None):
            train_model = Model(config, inputs, targets, batch_size, sequence_lengths=sequence_lengths,
                                is_training=True)

        sum_loss = tf.summary.scalar("loss", train_model.cross_entropy_loss)
        sum_val_loss = tf.summary.scalar("validation loss", train_model.cross_entropy_loss)
        sum_learn_rate = tf.summary.scalar("learning rate", train_model.learning_rate)

    merged = tf.summary.merge([sum_loss, sum_learn_rate])
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:

        session.run(init_op)

        train_dataset = config.train_dataset

        for i in range(config.epochs):

            train_dataset.shuffle()
            batches = train_dataset.partition(batch_size)
            # print("# of batches: %i" % len(batches))

            for j, (_inputs, _targets, _sequence_lengths, _) in enumerate(batches):

                feed_dict = {inputs: _inputs,
                             targets: _targets,
                             sequence_lengths: _sequence_lengths,
                             train_model.keep_prop: config.keep_prop}

                if j == 0:
                    if should_print:
                        print(i)
                    summary = session.run(merged, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, i)
                    if should_validate:
                        val_inputs, val_targets, val_lengths, _ = config.validation_dataset.partition(batch_size)[0]

                        val_feed_dict = {inputs: val_inputs,
                                         targets: val_targets,
                                         sequence_lengths: val_lengths,
                                         train_model.keep_prop: 1}

                        val_loss = session.run(sum_val_loss, feed_dict=val_feed_dict)
                        summary_writer.add_summary(val_loss, i)

                session.run(train_model.train_step, feed_dict=feed_dict)

        train_model.saver.save(session, "checkpoints/model.ckpt")


def test(config, summary_writer):
    batch_size = config.batch_size

    with tf.name_scope("Test"):
        inputs = tf.placeholder(tf.int32, shape=(None, batch_size), name="inputs")
        targets = tf.placeholder(tf.int32, shape=(None, batch_size), name="targets")
        sequence_lengths = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lengths")

        with tf.variable_scope("Model", reuse=True):
            test_model = Model(config, inputs, targets, batch_size, sequence_lengths=sequence_lengths)

        tf.summary.scalar("loss", test_model.cross_entropy_loss)

    with tf.Session() as session:
        test_model.saver.restore(session, "checkpoints/model.ckpt")

        test_dataset = config.test_dataset
        batches = test_dataset.partition(batch_size)

        predictions = []

        for _inputs, _targets, _sequence_lengths, names in batches:
            feed_dict = {inputs: _inputs,
                         targets: _targets,
                         sequence_lengths: _sequence_lengths}

            logits = test_model.logits
            loss = test_model.cross_entropy_loss

            # summary, out, loss = session.run([merged, logits, loss])
            out, loss = session.run([logits, loss], feed_dict=feed_dict)

            # np.set_printoptions(precision=5)
            batch_predictions = np.swapaxes(np.argmax(out, axis=2), 0, 1)
            batch_inputs = np.swapaxes(_inputs, 0, 1)
            batch_targets = np.swapaxes(_targets, 0, 1)

            for prediction in zip(names, _sequence_lengths, batch_inputs, batch_targets, batch_predictions):
                predictions.append(prediction)

                # print(loss)
                # print(np.swapaxes(np.array([pretty_out, pretty_targets]), 0, 2))

    return predictions
