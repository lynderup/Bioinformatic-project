
import tensorflow as tf
import itertools as it
import numpy as np
import lstm

import reader


num_seq = [1, 2, 4, 5, 6, 7, 8, 9, 10]
test_step_size = 2


def get_dummy_datasets(num_classes):

    _inputs = [[j for j in range(0, num_classes, i)] for i in num_seq]


    dummy_train_dataset = reader.Dataset(raw_data=np.array(_inputs), raw_labels=np.array(_inputs))

    test_input = np.array([[i for i in range(0, num_classes, test_step_size)]])
    test_targets = np.array([[i for i in range(0, num_classes, test_step_size)]])

    dummy_test_dataset = reader.Dataset(raw_data=test_input, raw_labels=test_targets)

    return dummy_train_dataset, dummy_test_dataset


class DummyModelConfig(object):
    num_units = 10
    num_classes = 100

    batch_size = len(num_seq)

    epochs = 200

    train_dataset, test_dataset = get_dummy_datasets(num_classes)


if __name__ == '__main__':
    config = DummyModelConfig()

    tf.reset_default_graph()
    summary_writer = tf.summary.FileWriter("logs/")

    lstm.train(config, summary_writer)
    lstm.test(config, summary_writer)

    summary_writer.add_graph(tf.get_default_graph())
    summary_writer.close()

    # dataset = reader.dataset160()
    # print(dataset.partition(10)[0])

