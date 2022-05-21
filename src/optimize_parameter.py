import os
import argparse
import json
import tensorflow as tf
import optuna
import pickle

from models.transformer import BinaryClassificationTransformer
from features.preprocessing import load_dataset


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('n_trials', type=int)
parser.add_argument('num_words', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('head_num', type=int)
parser.add_argument('val_threshold', type=float)

parser.add_argument('tfrecord_dir', type=str)
parser.add_argument('processed_dir', type=str)
parser.add_argument('study_path', type=str)
parser.add_argument('tuning_result_path', type=str)

args = parser.parse_args()


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    with open(args.study_path, 'wb') as f:
        pickle.dump(study, f)

    with open(args.tuning_result_path, 'w') as f:
        best_params = study.best_params
        print('best_params:', file=f)
        for key, val in best_params.items():
            print(f'{key}: {val}', file=f)

        print('\nbest_value: {}'.format(study.best_value), file=f)


def objective(trial):
    length = trial.suggest_int('length', 14, 30, step=2)
    lr = trial.suggest_float('lr', 10**(-6), 0.1, log=True)
    hidden_dim = 8 * trial.suggest_int('hidden_dim / 8', 4, 128, log=True)
    hopping_num = trial.suggest_int('hopping_num', 1, 6)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)

    # クラス重みを設定
    n_pos_neg_path = os.path.join(os.path.join(
        args.processed_dir, f'length{length}'), 'n_pos_neg.json')
    with open(n_pos_neg_path, 'r') as f:
        n_pos_neg = json.load(f)

    total = n_pos_neg['n_positive'] + n_pos_neg['n_negative']
    positive_weight = (1/n_pos_neg['n_positive']) * total / 2.0
    negative_weight = (1/n_pos_neg['n_negative']) * total / 2.0
    class_weight = {0: positive_weight, 1: negative_weight}

    # tfrecordの読み込み
    tfrecord_dir = os.path.join(args.tfrecord_dir, f'length{length}')
    train_tfrecord_path = os.path.join(tfrecord_dir, 'train_dataset.tfrecord')
    test_tfrecord_path = os.path.join(tfrecord_dir, 'test_dataset.tfrecord')

    train_ds = load_dataset(train_tfrecord_path,
                            batch_size=args.batch_size,
                            length=length+1)
    test_ds = load_dataset(test_tfrecord_path,
                            batch_size=args.batch_size,
                            length=length+1)

    # 学習
    model = create_model(lr=lr,
                         hopping_num=hopping_num,
                         hidden_dim=hidden_dim,
                         dropout_rate=dropout_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor='val_precision', mode='max', patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_precision', mode='max',
                factor=0.2, patience=3)
    ]
    history = model.fit(x=train_ds,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=test_ds,
                        callbacks=callbacks,
                        shuffle=True,
                        class_weight=class_weight,
                        verbose=1
                        )

    return max(history.history['val_precision'])

def create_model(lr, hopping_num, hidden_dim, dropout_rate):
    """ モデルを定義する """
    model = BinaryClassificationTransformer(
                vocab_size=args.num_words,
                hopping_num=hopping_num,
                head_num=args.head_num,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=[tf.keras.metrics.Precision(
                            thresholds=args.val_threshold,
                            name='precision'),
                          tf.keras.metrics.Recall(
                            thresholds=args.val_threshold,
                            name='recall')])

    return model


if __name__ == '__main__':
    main()