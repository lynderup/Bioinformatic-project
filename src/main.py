import os
import shutil
import tensorflow as tf
import numpy as np

import lstm
import reader
import compare_tm_pred as compare


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
    num_units = 10
    keep_prop = 0.5

    num_input_classes = 20
    num_output_classes = 3

    batch_size = 16

    epochs = 3


def do_run(config, should_test=False, should_print=False):
    tf.reset_default_graph()

    log_dir = "logs/"
    new_run_name = "".join((log_dir, "1"))

    if os.path.isdir(log_dir):
        _, runs, _ = list(os.walk(log_dir))[0]
        if len(runs) > 0:
            new_run_name = "".join((log_dir, str(max(map(int, runs)) + 1)))

    summary_writer = tf.summary.FileWriter(new_run_name)

    lstm.train(config, summary_writer, should_print=should_print)

    predictions = None
    if should_test:
        predictions = lstm.test(config, summary_writer)

    summary_writer.add_graph(tf.get_default_graph())
    summary_writer.flush()
    summary_writer.close()

    return predictions


def do_full_TMH_run():
    config = TMHModelConfig()
    config.train_dataset = reader.dataset160()

    do_run(config)


def do_TMH_fold_run(config, should_print=(False, False)):
    predictions = do_run(config, should_test=True, should_print=should_print[0])

    decoded_predictions = [reader.decode_example(prediction) for prediction in cut_to_lengths(predictions)]

    true, pred = to_dictionary(decoded_predictions)

    ac = compare.do_compare(true, pred, should_print[1])
    return ac


def do_TMH_10_fold():
    config = TMHModelConfig()
    datasets = reader.dataset160_10_fold()
    acs = []

    for i, (train_set, test_set) in enumerate(datasets):
        print("fold %i" % i)
        config.train_dataset = train_set
        config.test_dataset = test_set

        ac = do_TMH_fold_run(config, (True, False))
        acs.append(ac)

    mean = np.mean(acs)
    varians = np.var(acs)

    print("Mean:    %f" % mean)
    print("Varians: %f" % varians)
    print()

def do_first_TMH_fold():
    config = TMHModelConfig()
    datasets = reader.dataset160_10_fold()

    train_set, test_set = datasets.__next__()
    config.train_dataset = train_set
    config.test_dataset = test_set

    ac = do_TMH_fold_run(config, (True, True))

def cut_to_lengths(predictions):
    new_precictions = []
    for name, length, xs, zs, prediction in predictions:
        new_xs = xs[:length]
        new_zs = zs[:length]
        new_prediction = prediction[:length]

        new_precictions.append((name, new_xs, new_zs, new_prediction))

    return new_precictions


def to_dictionary(predictions):
    true = {}
    pred = {}

    for name, xs, zs, prediction in predictions:
        true[name] = "%s # %s" % (xs, zs)
        pred[name] = "%s # %s" % (xs, prediction)

    return true, pred


def clear_checkpoints():
    path = "checkpoints/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def clear_logs():
    path = "logs/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

if __name__ == '__main__':
    # config = DummyModelConfig()

    # clear_logs()
    clear_checkpoints()

    # do_TMH_run()
    # do_TMH_10_fold()
    do_first_TMH_fold()

    # dataset = reader.dataset160()
    # print(dataset.partition(10)[0])

