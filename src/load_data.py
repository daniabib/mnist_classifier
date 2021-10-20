import gzip
import numpy as np


def load_train_data():
    row_size = col_size = 28
    num_samples = 60_000

    f = gzip.open('../data/train-images-idx3-ubyte.gz', 'r')
    f.read(16)
    buf = f.read(row_size * col_size * num_samples)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return data.reshape((num_samples, row_size, col_size))


def load_test_data():
    row_size = col_size = 28
    num_samples = 10_000

    f = gzip.open('../data/t10k-images-idx3-ubyte.gz', 'r')
    f.read(16)
    buf = f.read(row_size * col_size * num_samples)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return data.reshape((num_samples, row_size, col_size))


def load_train_labels():
    f = gzip.open('../data/train-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    buf = f.read()
    return np.frombuffer(buf, dtype=np.dtype(np.uint8).newbyteorder(">"))


def load_test_labels():
    f = gzip.open('../data/t10k-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    buf = f.read()
    return np.frombuffer(buf, dtype=np.dtype(np.uint8).newbyteorder(">"))


def load_data():
    return load_train_data(), load_test_data(), load_train_labels(), load_test_labels()


if __name__ == '__main__':
    train_data, test_data, train_label, test_label = load_data()
    print(train_data.shape)
    print(test_data.shape)
    print(train_label.shape)
    print(test_label.shape)
