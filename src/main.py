
import tensorflow as tf
import itertools as it
import numpy as np

test1 = [i for i in range(0, 10, 1)]
test2 = [i for i in range(0, 10, 1)]


inputs = [test1, test2]

with tf.name_scope("inputs"):
    tf_inputs = [tf.convert_to_tensor([[i1], [i2]], dtype=tf.float32) for i1, i2 in it.zip_longest(*inputs, fillvalue=0)]

# tf.summary.tensor_summary('inputs_sum', tf_inputs)

sequence_lengths = [len(i) for i in inputs]


cell = tf.contrib.rnn.BasicLSTMCell(num_units=1, forget_bias=0.0, state_is_tuple=True)

with tf.name_scope("initial_state"):
    initial_state = cell.zero_state(batch_size=2, dtype=tf.float32)
    # tf.summary.tensor_summary('initial_state', initial_state)

outputs, state = tf.contrib.rnn.static_rnn(cell, tf_inputs, initial_state=initial_state, scope="rnn", sequence_length=sequence_lengths)


# merged = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init_op)
    # print(session.run([initial_state, initial_state_batch]))
    # print(session.run([state, state_batch]))
    out = session.run(outputs)

    # np.set_printoptions(precision=5)
    print(np.array([i.tolist() for i in out]))

    summary_writer = tf.summary.FileWriter("logs/", session.graph)
    # summary_writer.add_summary(summary)

