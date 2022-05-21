import os
import argparse
import numpy as np
import pickle
import json
import math
import re
import tensorflow as tf

from features.preprocessing import Vocab, add_class_token, underSampling


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('processed_dir', type=str)
parser.add_argument('tfrecord_path', type=str)
parser.add_argument('vocab_path', type=str)
parser.add_argument('processed_dir', type=str)
parser.add_argument('--val_rate', type=float, default=0.2)

args = parser.parse_args()


def main():
    json_path = 'references/PTAP_data.json'
    with open(json_path, 'r') as f:
       json_data = json.load(f)

    with open(args.vocab_path, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab = Vocab(tokenizer)

    match = re.search(r'\d+', args.tfrecord_path)
    length = int(match.group())

    x_train, y_train, x_test, y_test = [], [], [], []
    for data in json_data:
        virusname = data['virus'].replace(' ', '_')

        # アミノ酸断片データセットを読み込み
        processed_path = os.path.join(os.path.join(
            args.processed_dir, f'length{length}'), f'{virusname}.pickle')
        with open(processed_path, 'rb') as f:
            x = pickle.load(f)
            y = pickle.load(f)

        np.random.seed(1)
        np.random.shuffle(x)
        np.random.seed(1)
        np.random.shuffle(y)

        # データセットを学習用と検証用に分割
        boundary = math.floor(len(x) * args.val_rate)
        x_test.append(x[:boundary])
        y_test.append(y[:boundary])
        x_train.append(x[boundary:])
        y_train.append(y[boundary:])

    for i, data in enumerate(json_data):
        x_train[i] = vocab.encode(x_train[i])
        x_test[i]  = vocab.encode(x_test[i])
        x_train[i] = add_class_token(x_train[i])
        x_test[i] = add_class_token(x_test[i])

    x_test, x_train = np.vstack(x_test), np.vstack(x_train)
    y_test, y_train = np.hstack(y_test), np.hstack(y_train)

    # undersamplingの比率の計算(陰性を1/10に減らす)
    n_positive = (y_train == 1).sum()
    n_negative = (len(y_train) - n_positive) // 10
    sampling_strategy = n_positive / n_negative

    # 陽性・陰性のサンプル数をjsonファイルとして記憶
    # 学習で class weight を計算するときに利用する
    dict = {'n_positive': int(n_positive), 'n_negative': int(n_negative)}
    n_pos_neg_path = os.path.join(args.processed_dir,
            f'length{length}/n_pos_neg.json')
    with open(n_pos_neg_path, 'w') as f:
        json.dump(dict, f)

    # undersampling
    x_train, y_train = underSampling(x_train, y_train,
                            sampling_strategy=sampling_strategy)
    x_test, y_test = underSampling(x_test, y_test,
                            sampling_strategy=1.0)

    # シャッフル
    np.random.seed(1)
    np.random.shuffle(x_test)
    np.random.seed(1)
    np.random.shuffle(y_test)
    np.random.seed(1)
    np.random.shuffle(x_train)
    np.random.seed(1)
    np.random.shuffle(y_train)

    # tf.data.Datasetに変換
    tfrecord_dir = os.path.dirname(args.tfrecord_path)
    test_tfrecord_path = os.path.join(tfrecord_dir,
                                      'test_dataset.tfrecord')
    train_tfrecord_path = os.path.join(tfrecord_dir,
                                       'train_dataset.tfrecord')
    write_tfrecord(x_test, y_test, test_tfrecord_path)
    write_tfrecord(x_train, y_train, train_tfrecord_path)


def make_example(sequence, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(
                int64_list=tf.train.Int64List(value=sequence)),
        'y': tf.train.Feature(
                int64_list=tf.train.Int64List(value=label))
    }))

def write_tfrecord(sequences, labels, filename):
    """ tf.data.Datasetに変換 """
    writer = tf.io.TFRecordWriter(filename)
    for sequence, label in zip(sequences, labels):
        ex = make_example(sequence.tolist(), [int(label)])

        writer.write(ex.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
