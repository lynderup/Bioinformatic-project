import os
import tensorflow as tf
import numpy as np

import lstm
import reader
import compare_tm_pred as compare


num_seq = [1, 2, 4, 5, 6, 7, 8, 9, 10]
test_step_size = 3


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

class TMHModelConfig(object):
    num_units = 10
    num_classes = 20

    batch_size = 10

    epochs = 50

    train_dataset = reader.dataset160()


if __name__ == '__main__':
    # config = DummyModelConfig()
    config = TMHModelConfig()

    tf.reset_default_graph()

    log_dir = "logs/"
    new_run_name = "".join((log_dir, "1"))

    _, runs, _ = list(os.walk(log_dir))[0]
    if len(runs) > 0:
        new_run_name = "".join((log_dir, str(int(max(runs)) + 1)))

    summary_writer = tf.summary.FileWriter(new_run_name)

    lstm.train(config, summary_writer)
    # lstm.test(config, summary_writer)

    summary_writer.add_graph(tf.get_default_graph())
    summary_writer.close()

    # dataset = reader.dataset160()
    # print(dataset.partition(10)[0])

