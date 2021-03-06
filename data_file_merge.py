import os
import xarray
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

def merge_nc_files(attr):
    basefile = attr[0]
    savepath = os.path.join(basefile, attr[1])
    var = attr[2]
    urls = sorted(os.listdir(basefile))
    urls = [os.path.join(basefile, i) for i in urls]
    datasets = [xarray.open_dataset(url, cache=True, decode_times=False)[var] for url in urls]
    merged = xarray.concat(datasets, 'time')
    merged.to_netcdf(savepath)


def read_data():
    file = os.path.join(f'{hp.observe_dataset_dir}/meta-data/wind', 'vwind.mon.mean1980-2019.nc')
    sea_data = xarray.open_dataset(file, cache=True, decode_times=True)
    print(sea_data)
    # print(sea_data.sst)


def main():
    for i in dic:
        merge_nc_files(i)


if __name__ == "__main__":
    # read_data()
    main()
