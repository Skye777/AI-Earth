import os
import json
import random
import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from progress.bar import PixelBar

from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


# ---------- Helpers ----------
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# ---------- Prepare Data ----------
def parse_npz_and_nc_data():
    height = hp.height
    width = hp.width

    sst = np.load(f"{hp.npz_dir}/soda_sst.npz")['sst']
    t300 = np.load(f"{hp.npz_dir}/soda_t300.npz")['t300']
    ua = np.load(f"{hp.npz_dir}/soda_ua.npz")['ua']
    va = np.load(f"{hp.npz_dir}/soda_va.npz")['va']

    # sst[abs(sst) < 8e-17] = 0

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = Normalizer()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, height * width))), (-1, height, width))
    t300 = np.reshape(scaler.fit_transform(np.reshape(t300, (-1, height * width))), (-1, height, width))
    ua = np.reshape(scaler.fit_transform(np.reshape(ua, (-1, height * width))), (-1, height, width))
    va = np.reshape(scaler.fit_transform(np.reshape(va, (-1, height * width))), (-1, height, width))

    data = []
    target = []
    for i in range(sst.shape[0] - hp.in_seqlen + 1 - hp.lead_time - hp.out_seqlen):
        data.append({'sst': sst[i:i + hp.in_seqlen].astype(np.float32),
                     't300': t300[i:i + hp.in_seqlen].astype(np.float32),
                     'ua': ua[i:i + hp.in_seqlen].astype(np.float32),
                     'va': va[i:i + hp.in_seqlen].astype(np.float32)})
        
        target_start = i + hp.in_seqlen - 1 + hp.lead_time
        target.append({'sst': sst[target_start:target_start + hp.out_seqlen].astype(np.float32),
                       't300': t300[target_start:target_start + hp.out_seqlen].astype(np.float32),
                       'ua': ua[target_start:target_start + hp.out_seqlen].astype(np.float32),
                       'va': va[target_start:target_start + hp.out_seqlen].astype(np.float32)})

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=hp.train_eval_split,
                                                                        random_state=hp.random_seed)
    print(len(train_data), len(test_data), len(train_target), len(test_target))
    return train_data, test_data, train_target, test_target


# ---------- IO ----------
def write_records(data, filename):
    series = data[0]
    target = data[1]
    writer = tf.io.TFRecordWriter(f'{hp.preprocess_out_dir}/{filename}')

    bar = PixelBar(r'Generating', max=len(data), suffix='%(percent)d%%')
    for s, t in zip(series, target):
        example = tf.train.Example(features=tf.train.Features(feature={
            'input_sst': _bytes_feature(s['sst'].tobytes()),
            'input_t300': _bytes_feature(s['t300'].tobytes()),
            'input_ua': _bytes_feature(s['ua'].tobytes()),
            'input_va': _bytes_feature(s['va'].tobytes()),
            'output_sst': _bytes_feature(t['sst'].tobytes()),
            'output_t300': _bytes_feature(t['t300'].tobytes()),
            'output_ua': _bytes_feature(t['ua'].tobytes()),
            'output_va': _bytes_feature(t['va'].tobytes()),
        }))
        writer.write(example.SerializeToString())
        bar.next()
    writer.close()
    bar.finish()


# ---------- Go! ----------
if __name__ == "__main__":
    if not os.path.exists(hp.preprocess_out_dir):
        print("Creating output directory {}...".format(hp.preprocess_out_dir))
        os.makedirs(hp.preprocess_out_dir)

    print("Parsing raw data...")
    train_data, test_data, train_target, test_target = parse_npz_and_nc_data()
    print("Writing TF Records to file...")
    write_records((train_data, train_target), "train.tfrecords")
    write_records((test_data, test_target), "test.tfrecords")

    print("Done!")
