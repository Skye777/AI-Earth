"""
@author: Skye Cui
@file: tester.py
@time: 2021/3/22 10:32
@description: 
"""
import os
import shutil
import numpy as np
import tensorflow as tf
from model import UTransformer
from hparams import Hparams
from loss import Loss
import zipfile

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def nino_seq(ssta):
    # inputs: [24, h, w]
    nino = []
    n_index = [np.mean(ssta[i, 10:13, 38:49]) for i in range(len(ssta))]
    nino.append(n_index)
    return nino


def test(in_path='./tcdata/enso_round1_test_20210201/',
         out_path='result'):
    if not os.path.exists(in_path):
        for path in hp.path_list:
            if os.path.exists(path):
                in_path = path

    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)

    test_sample_file = [os.path.join(in_path, i) for i in os.listdir(in_path) if i.endswith('.npy')]
    model = UTransformer(hp)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.lr)
    model_loss = Loss(model)
    model.compile(optimizer, model_loss)
    x = np.random.random((4, 12, 24, 72, 4))
    ys = (np.random.random((4, 12, 24, 72, 4)), np.random.random((4, 12, 24, 72, 4)))
    model.train_on_batch([x, ys])
    # model = tf.keras.models.load_model(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')
    model.load_weights(f'{hp.delivery_model_dir}/{hp.delivery_model_file}')

    for i in test_sample_file:
        data = np.load(i)
        data = np.nan_to_num(data)

        preds = model(data, data, training=False)
        nino_index = nino_seq(preds[:, :, :, 0])

        save_path = os.path.join(out_path, os.path.basename(i))
        np.save(file=save_path, arr=nino_index)
    make_zip(out_path, 'result.zip')


def make_zip(source_dir, output_filename):
    f = zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            f.write(os.path.join(dirpath, filename))
    f.close()


if __name__ == '__main__':
    test()

