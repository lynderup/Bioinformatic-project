import numpy as np

import reader


def get_dummy_datasets(num_classes, num_seq, test_step_size):

    _inputs = [[j for j in range(0, num_classes, i)] for i in num_seq]

    dummy_train_dataset = reader.Dataset(raw_data=np.array(_inputs), raw_labels=np.array(_inputs))

    test_input = np.array([[i for i in range(0, num_classes, test_step_size)]])
    test_targets = np.array([[i for i in range(0, num_classes, test_step_size)]])

    dummy_test_dataset = reader.Dataset(raw_data=test_input, raw_labels=test_targets)

    return dummy_train_dataset, dummy_test_dataset


class DummyModelConfig(object):
    num_seq = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    test_step_size = 3

    num_units = 10

    num_input_classes = 100
    num_output_classes = 100

    batch_size = len(num_seq)

    epochs = 200

    train_dataset, test_dataset = get_dummy_datasets(num_input_classes, num_seq, test_step_size)


class TMHModelConfig(object):
    num_units = 20
    keep_prop = 0.5
    l2_reg = False

    num_input_classes = 20
    num_output_classes = 3

    batch_size = 16

    starting_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.96

    epochs = 50
