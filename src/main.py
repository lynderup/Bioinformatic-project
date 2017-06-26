import os
import shutil
import tensorflow as tf

import lstm
import reader
import write_prediction as writer
import model_configs as models
import statistics as stat

import reduce_noise


def do_run(config, should_test=False, should_validate=False, should_print=False):
    tf.reset_default_graph()

    log_dir = "logs/"
    new_run_name = "".join((log_dir, "1"))

    if os.path.isdir(log_dir):
        _, runs, _ = list(os.walk(log_dir))[0]
        if len(runs) > 0:
            new_run_name = "".join((log_dir, str(max(map(int, runs)) + 1)))

    summary_writer = tf.summary.FileWriter(new_run_name)

    lstm.train(config, summary_writer, new_run_name, should_print=should_print, should_validate=should_validate)

    predictions = None
    if should_test:
        predictions = lstm.test(config, summary_writer, new_run_name)

    summary_writer.add_graph(tf.get_default_graph())
    summary_writer.flush()
    summary_writer.close()

    return predictions


def do_full_TMH_run():
    config = models.TMHModelConfig()
    config.train_dataset = reader.dataset160()

    do_run(config)


def do_TMH_fold_run(config, stat_wn, stat_won, filename, should_print=(False, False)):
    predictions = do_run(config, should_test=True, should_validate=True, should_print=should_print[0])

    decoded_predictions = [reader.decode_example(prediction) for prediction in cut_to_lengths(predictions)]
    noise_reduced = reduce_noise.reduce_noise(decoded_predictions)

    stat_wn.add_prediction(decoded_predictions)
    stat_won.add_prediction(noise_reduced)

    if filename is not None:
        writer.write_predictions(decoded_predictions, filename)
        writer.write_predictions(noise_reduced, "%s_noise_reduced" % filename)


def do_TMH_10_fold():
    config = models.TMHModelConfig()
    datasets = reader.dataset160_10_fold()

    lstm_stat = stat.Statistic("LSTM")
    noise_reduced_lstm_stat = stat.Statistic("Noise reduced LSTM")

    for i, (train_set, test_set) in enumerate(datasets):
        print("fold %i" % i)
        config.train_dataset = train_set
        config.test_dataset = test_set
        config.validation_dataset = test_set

        do_TMH_fold_run(config, lstm_stat, noise_reduced_lstm_stat, "fold%i" % i, (True, False))

    lstm_stat.print_statistics()
    noise_reduced_lstm_stat.print_statistics()


def do_first_TMH_fold():
    config = models.TMHModelConfig()
    datasets = reader.dataset160_10_fold()

    train_set, test_set = datasets.__next__()
    config.train_dataset = train_set
    config.test_dataset = test_set
    config.validation_dataset = test_set

    lstm_stat = stat.Statistic("LSTM")
    noise_reduced_lstm_stat = stat.Statistic("Noise reduced LSTM")

    do_TMH_fold_run(config, lstm_stat, noise_reduced_lstm_stat, "test", (True, True))

    lstm_stat.print_statistics()
    noise_reduced_lstm_stat.print_statistics()


def cut_to_lengths(predictions):
    new_precictions = []
    for name, length, xs, zs, prediction in predictions:
        new_xs = xs[:length]
        new_zs = zs[:length]
        new_prediction = prediction[:length]

        new_precictions.append((name, new_xs, new_zs, new_prediction))

    return new_precictions


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
    # clear_checkpoints()

    do_TMH_10_fold()

    # do_first_TMH_fold()

    # dataset = reader.dataset160()
    # print(dataset.partition(10)[0])

