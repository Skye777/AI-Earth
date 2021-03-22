"""
@author: Skye Cui
@file: data_read.py
@time: 2021/3/17 18:50
@description: 
"""
import os
import xarray
import numpy as np
import matplotlib.pyplot as plt

# Dimensions:  (month: 36, year: 4645)
# Coordinates:
#   * year     (year) int32 1 2 3 4 5 6 7 8 ... 4639 4640 4641 4642 4643 4644 4645
#   * month    (month) int32 1 2 3 4 5 6 7 8 9 10 ... 28 29 30 31 32 33 34 35 36
# Data variables:
#     nino     (year, month) float64 ...

# Dimensions:  (lat: 24, lon: 72, month: 36, year: 4645)
# Coordinates:
#   * year     (year) int32 1 2 3 4 5 6 7 8 ... 4639 4640 4641 4642 4643 4644 4645
#   * month    (month) int32 1 2 3 4 5 6 7 8 9 10 ... 28 29 30 31 32 33 34 35 36
#   * lat      (lat) float32 -55.0 -50.0 -45.0 -40.0 -35.0 ... 45.0 50.0 55.0 60.0
#   * lon      (lon) float32 0.0 5.0 10.0 15.0 20.0 ... 340.0 345.0 350.0 355.0
# Data variables:
#     sst      (year, month, lat, lon) float64 ...
#     t300     (year, month, lat, lon) float64 ...
#     ua       (year, month, lat, lon) float64 ...
#     va       (year, month, lat, lon) float64 ...

# Dimensions:  (lat: 24, lon: 72, month: 36, year: 100)
# Coordinates:
#   * year     (year) int32 1 2 3 4 5 6 7 8 9 10 ... 92 93 94 95 96 97 98 99 100
#   * month    (month) int32 1 2 3 4 5 6 7 8 9 10 ... 28 29 30 31 32 33 34 35 36
#   * lat      (lat) float64 -55.0 -50.0 -45.0 -40.0 -35.0 ... 45.0 50.0 55.0 60.0
#   * lon      (lon) float64 0.0 5.0 10.0 15.0 20.0 ... 340.0 345.0 350.0 355.0
# Data variables:
#     sst      (year, month, lat, lon) float32 ...
#     t300     (year, month, lat, lon) float32 ...
#     ua       (year, month, lat, lon) float64 ...
#     va       (year, month, lat, lon) float64 ...

# Dimensions:  (month: 36, year: 100)
# Coordinates:
#   * year     (year) int32 1 2 3 4 5 6 7 8 9 10 ... 92 93 94 95 96 97 98 99 100
#   * month    (month) int32 1 2 3 4 5 6 7 8 9 10 ... 28 29 30 31 32 33 34 35 36
# Data variables:
#     nino     (year, month) float64 ...

meta_data = 'D:\Python\AI-Earth\meta_data'
final_data = '/home/dl/Public/Skye/AI-Earth/final_data'


def nino_seq():
    data = os.path.join(meta_data, 'SODA_train.nc')
    ssta = xarray.open_dataset(data, cache=True, decode_times=True)['sst']
    print(ssta.shape)
    nino = []
    for sample in range(len(ssta)):
        n_index = [np.mean(ssta[sample, i,  10:13, 38:49]) for i in range(len(ssta[sample]))]
        print(np.array(n_index).shape)
        nino.append(n_index)
    print(np.array(nino).shape)


def nino34_index():
    train_data = os.path.join(meta_data, 'SODA_train.nc')
    label_data = os.path.join(meta_data, 'SODA_label.nc')
    ssta = xarray.open_dataset(train_data, cache=True, decode_times=True)['sst']
    label = xarray.open_dataset(label_data, cache=True, decode_times=True)['nino']
    # print(label[0].values)

    true_label = []
    for i in range(100):
        true_label.extend(label[i, :].values)
    l1 = plt.plot(true_label, 'r--')
    print(np.array(true_label).shape)

    la = ssta.coords["lat"]
    lc = ssta.coords["lon"]
    nino_ssta = ssta.loc[dict(lat=la[(la >= -5) & (la <= 5)], lon=lc[(lc >= 190) & (lc <= 240)])]
    ssta_seq = []
    for i in range(100):
        ssta_seq.extend(nino_ssta[i, :].values)

    cal_label = []
    for i in range(len(ssta_seq)-2):
        ssta1 = np.mean(ssta_seq[i])
        ssta2 = np.mean(ssta_seq[i + 1])
        ssta3 = np.mean(ssta_seq[i + 2])
        nino3_4 = (ssta1 + ssta2 + ssta3) / 3
        cal_label.append(nino3_4)
    print(np.array(cal_label).shape)
    # print(nino)
    l2 = plt.plot(cal_label, 'b--')
    plt.show()


def visualize():
    file = os.path.join(meta_data, 'SODA_label.nc')
    data = xarray.open_dataset(file, cache=True, decode_times=True)
    print(data)
    la = data.coords["lat"]
    lc = data.coords["lon"]
    sample = data.sel(year=1, month=1).sst
    nino34 = sample.loc[dict(lat=la[(la >= -5) & (la <= 5)], lon=lc[(lc >= 190) & (lc <= 240)])]
    nino34.plot()
    plt.show()


def npz_data():
    cmip_sst = np.load(f"{final_data}/{'cmip_sst'}.npz")['sst'][:15000]
    # vwind = np.load(f"{dir_path}/{'vwind-resolve'}.npz")['vwind']
    # sst = np.load(f"{dir_path}/{'sst-resolve'}.npz")['sst']
    # sshg = np.load(f"{dir_path}/{'sshg'}.npz")['sshg']
    # thflx = np.load(f"{dir_path}/{'thflx'}.npz")['thflx']

    print(cmip_sst.shape)


if __name__ == "__main__":
    # visualize()
    # nino34_index()
    npz_data()
    # nino_seq()
