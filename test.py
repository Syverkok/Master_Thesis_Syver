from pydap.client import open_url
from datetime import datetime
import numpy as np
import pandas as pd
import time
import pyproj
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import math
import pickle
from datetime import date
import os


def reduce_area_of_df(df):
    df = df[df.sp_lat <= 9.5]
    df = df[df.sp_lat >= 0]
    df = df[df.sp_lon <= -85.5]
    return df[df.sp_lon >= -100.5]


def generate_url(year, month, day, satellite_number):
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url
    clickable_url = base_url + specific_url + '.html'

    return data_url + '?sp_lat,sp_lon,track_id,quality_flags,ddm_timestamp_utc,ddm_nbrcs', clickable_url


def store_cygnss_data(year, month, day):
    sat_df = pd.DataFrame()
    for sat_numb in range(1, 9):  # Remember to change back to  1, 9
        print("Satellite number : " + str(sat_numb))
        test_data_url, test_clickable_url = generate_url(year, month, day, sat_numb)
        dataset = open_url(test_data_url, output_grid=False)
        for ddm in range(4):  # Remember to change back to 4
            ddm_df = pd.DataFrame()
            print("ddm : " + str(ddm))
            sp_lat = np.array(dataset.sp_lat[:, ddm])
            sp_lon = np.array(dataset.sp_lon[:, ddm])
            a, b = (np.where(sp_lon > 180))
            sp_lon[a] -= 360

            ddm_timestamp_utc = np.array(dataset.ddm_timestamp_utc[:, ddm])
            d0 = date(1992, 10, 5)
            d1 = date(year, month, day)
            delta = d1 - d0
            hours = delta.days * 24
            ddm_timestamp_utc = np.rint(ddm_timestamp_utc / 3600) + hours

            ddm_nbrcs = np.array(dataset.ddm_nbrcs[:, ddm])

            ddm_df['sp_lat'] = sp_lat.tolist()
            ddm_df['sp_lon'] = sp_lon.tolist()
            '''
            track_id = np.array(dataset.track_id[:, ddm])
            quality_flags = np.array(dataset.quality_flags[:, ddm])
            ddm_df['ddm_channel'] = np.zeros(len(sp_lon))
            ddm_df = ddm_df.assign(ddm_channel=ddm)
            ddm_df['track_id'] = track_id.tolist()
            ddm_df['quality_flags'] = quality_flags.tolist()
            '''
            ddm_df['hours_since_ref'] = ddm_timestamp_utc.tolist()
            ddm_df['ddm_nbrcs'] = ddm_nbrcs.tolist()

            for col in ddm_df.columns:
                if col != 'ddm_channel' and col != 'hours_since_ref':
                    ddm_df[col] = ddm_df[col].apply(lambda x: x[0])
            sat_df = sat_df.append(ddm_df, ignore_index=True)
    sat_df.to_pickle('df.pkl')
    return sat_df


def open_oskar_data_local(filename):
    ds = xr.open_dataset(filename)
    oskar_df = ds.to_dataframe()
    oskar_df.dropna(inplace=True)
    oskar_df = oskar_df.reset_index()
    d0 = date(1992, 10, 5)
    d1 = oskar_df['time'][1].date()
    delta = d1 - d0
    hours = delta.days * 24
    oskar_df['time'] = np.zeros(len(oskar_df['time']))
    oskar_df.assign(time=hours)
    oskar_df = oskar_df.rename(columns={"latitude": "sp_lat", "longitude": "sp_lon", "time": "hours_since_ref"})
    # oskar_df = reduce_area_of_df(oskar_df)
    return oskar_df


list_of_dfs = []
for filename in os.listdir('oskar_data'):
    list_of_dfs.append(open_oskar_data_local('oskar_data/' + filename))
oskar_df = pd.concat(list_of_dfs)
print(min(oskar_df['sp_lon'].unique()))
