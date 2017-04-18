import itertools as it
import numpy as np

def read_sequence(file_part):
    parts = file_part.split("\n")

    name = parts[0][1:]

    sequences = "".join("".join(parts[1:]).split()).split("#")
    return name, sequences[0], sequences[1]


def read_datasets(filenames):
    datasets = []

    for filename in filenames:
        with open(filename, "r") as file:

            file_parts = file.read().split("\n\n")

            test_sequences = []

            for file_part in file_parts:
                if len(file_part) == 0:
                    continue
                if file_part[0] != ">":
                    continue

                test_sequences.append(read_sequence(file_part))

            datasets.append(test_sequences)

    return datasets


def encode_example(example):
    hidden = ['i', 'M', 'o']
    observables = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']

    name, xs, zs = example

    new_xs = [observables.index(x) for x in xs]
    new_zs = [hidden.index(z) for z in zs]

    return name, new_xs, new_zs


def dataset160():
    path_to_data = "data/Dataset160/"
    dataset_files = ["set160.%i.labels.txt" % i for i in range(10)]

    datasets = read_datasets([path_to_data + file for file in dataset_files])

    datasets = [[encode_example(example) for example in dataset] for dataset in datasets]

    raw_data = []
    raw_labels = []

    for dataset in datasets:
        for name, xs, zs in dataset:
            raw_data.append(xs)
            raw_labels.append(zs)

    return Dataset(np.array(raw_data), np.array(raw_labels))


def dataset160_10_fold():
    path_to_data = "data/Dataset160/"
    dataset_files = ["set160.%i.labels.txt" % i for i in range(10)]

    datasets = read_datasets([path_to_data + file for file in dataset_files])

    datasets = [[encode_example(example) for example in dataset] for dataset in datasets]

    train_raw_data = []
    train_raw_labels = []

    test_raw_data = []
    test_raw_labels = []

    i = 0
    while i < 10:
        for idx, dataset in enumerate(datasets):
            for name, xs, zs in dataset:
                if idx == i:
                    test_raw_data.append(xs)
                    test_raw_labels.append(zs)
                else:
                    train_raw_data.append(xs)
                    train_raw_labels.append(zs)

        train_set = Dataset(np.array(train_raw_data), np.array(train_raw_labels))
        test_set = Dataset(np.array(test_raw_data), np.array(test_raw_labels))

        yield train_set, test_set
        i += 1


class Dataset:

    def __init__(self, raw_data, raw_labels):

        self.num_examples = len(raw_data)
        self.raw_data = raw_data
        self.raw_labels = raw_labels

    def shuffle(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)

        self.raw_data = self.raw_data[perm]
        self.raw_labels = self.raw_labels[perm]

    def partition(self, batch_size):

        batch = []

        for i in range(batch_size, self.num_examples+1, batch_size):
            start = i - batch_size
            end = i

            batch_data = self.raw_data[start:end]
            batch_labels = self.raw_labels[start:end]

            _input = np.array(list(it.zip_longest(*batch_data, fillvalue=0)))
            target = np.array(list(it.zip_longest(*batch_labels, fillvalue=0)))
            lengths = [len(xs) for xs in batch_data]

            batch.append((_input, target, lengths))

        return batch



