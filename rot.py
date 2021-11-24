from pydap.client import open_url
from datetime import datetime
import numpy as np
import pandas as pd
import time
from datetime import timedelta
import pyproj
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import math
import pickle
from datetime import date
import os
import plotly.express as px
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
import cartopy as cart


def retrieve_microplastics_gt():
    lats = np.zeros(181)
    count = 90
    for i in range(len(lats)):
        lats[i] = count
        count -= 1
    for filename in os.listdir('microplast_gt'):
        logfilename = filename[:-4] + '_log'
        try:
            df = pd.read_csv('microplast_gt/' + filename, header=None)
        except:
            print(filename)
            continue
        longlist = []
        latlist = []
        valuelist_log = []
        valuelist = []
        for long in range(0, 361):
            for lat in range(0, 181):
                val = df[long][lat]
                if val < 1:
                    val = np.nan
                longlist.append(long)
                latlist.append(lats[lat])
                valuelist_log.append(np.log10(val))
                valuelist.append(val)
        if filename == "lebretonmodel_abundance.csv":
            res_df = pd.DataFrame(
                {'sp_lon': longlist, 'sp_lat': latlist, filename[:-4]: valuelist, logfilename: valuelist_log})
        else:
            res_df[filename[:-4]] = valuelist
            res_df[logfilename] = valuelist_log
    return res_df


def plot_wind(df, var1, var2, interpolated=False):
    # Settings for the plot
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.gridlines(draw_labels=True, alpha=0.5)
    plt.scatter(df['sp_lon'], df['sp_lat'], c=list(np.sqrt(df[var1] ** 2 + df[var2] ** 2)), cmap='RdBu')
    bar = plt.colorbar(pad=0.15, orientation='horizontal')
    bar.ax.set_title('m/s')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if interpolated:
        plt.savefig('wind_speed_inter.png')
    else:
        plt.savefig('wind_speed.png')
    plt.show()


def plot_var(df, var):
    # Settings for the plot
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    ax.gridlines(draw_labels=True, alpha=0.5)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.scatter(df['sp_lon'], df['sp_lat'], c=list(df[var]))
    bar = plt.colorbar(pad=0.15, orientation='horizontal')
    plt.title(var)
    bar.ax.set_title('Power 10^')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('nbrcs-diff.svg')
    plt.show()


def generate_url(year, month, day, satellite_number):
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url
    clickable_url = base_url + specific_url + '.html'

    return data_url + '?sp_lat,sp_lon,track_id,prn_code,quality_flags,ddm_timestamp_utc,ddm_nbrcs,fresnel_coeff,sp_inc_angle', clickable_url


def fetch_cygnss(y1, m1, d1, y2, m2, d2):
    sdate = date(y1, m1, d1)  # start date
    edate = date(y2, m2, d2)  # end date
    delta = edate - sdate  # as timedelta
    df_list = []
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        df = get_cygnss_data(day.year, day.month, day.day)
        df = prep_cygnss(df)
        df_list.append(df)
        df.to_csv("cygnss_data/" + str(day.year) + str(day.month) + str(day.day) + ".csv", index=False)
    if len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def hours_since_ref(year, month, day):
    d0 = date(1992, 10, 5)
    d1 = date(year, month, day)
    delta = d1 - d0
    hours = delta.days * 24
    return hours


def get_cygnss_data(year, month, day):
    cygnss_df = pd.DataFrame()
    hours = hours_since_ref(year, month, day)
    for sat_numb in range(1, 9):  # Remember to change back to  1, 9
        print("Satellite number : " + str(sat_numb))
        test_data_url, test_clickable_url = generate_url(year, month, day, sat_numb)
        dataset = open_url(test_data_url, output_grid=False)
        for ddm in range(4):  # Remember to change back to 4
            ddm_df = pd.DataFrame()
            print("ddm : " + str(ddm))

            ddm_timestamp_utc = np.array(dataset.ddm_timestamp_utc[:, ddm])
            ddm_timestamp_utc = np.rint(ddm_timestamp_utc / 3600) + hours

            ddm_df['sp_lat'] = np.array(dataset.sp_lat[:, ddm]).tolist()
            sp_lon = np.array(dataset.sp_lon[:, ddm]).tolist()
            ddm_df['sp_lon'] = sp_lon
            ddm_df['sp_lon'] = np.array(dataset.sp_lon[:, ddm]).tolist()
            ddm_df['hours_since_ref'] = ddm_timestamp_utc.tolist()
            ddm_df['ddm_nbrcs'] = np.array(dataset.ddm_nbrcs[:, ddm]).tolist()
            ddm_df['fresnel_coeff'] = np.array(dataset.fresnel_coeff[:, ddm]).tolist()
            ddm_df['sp_inc_angle'] = np.array(dataset.sp_inc_angle[:, ddm]).tolist()
            ddm_df['quality_flags'] = np.array(dataset.quality_flags[:, ddm]).tolist()
            ddm_df['track_id'] = np.array(dataset.track_id[:, ddm]).tolist()

            for col in ddm_df.columns:
                if col != 'ddm_channel' and col != 'hours_since_ref' and col != 'unique_track_id':
                    ddm_df[col] = ddm_df[col].apply(lambda x: x[0])
            hoursince = [str(hours)] * len(sp_lon)
            sat_numb_ar = [str(sat_numb)] * len(sp_lon)
            ddm_df['unique_track_id'] = generate_unique_track_id_value(hoursince, list(map(str, ddm_df['track_id'])),
                                                                       sat_numb_ar)
            cygnss_df = cygnss_df.append(ddm_df, ignore_index=True)
    return cygnss_df


def generate_unique_track_id_value(hoursince, track_id, sat_nr):
    return list(map(''.join, zip(*[hoursince, track_id, sat_nr])))


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
    oskar_df = oskar_df.assign(time=hours)
    oskar_df = oskar_df.rename(columns={"latitude": "sp_lat", "longitude": "sp_lon", "time": "hours_since_ref"})
    oskar_df = oskar_df.loc[oskar_df['sp_lon'] < 380]
    oskar_df['sp_lon'] = oskar_df['sp_lon'] % 360
    oskar_df = reduce_area_of_df(oskar_df)
    return oskar_df


def fetch_all_oskar_files():
    df_list = []
    for filename in os.listdir('oskar_data'):
        df_list.append(open_oskar_data_local('oskar_data/' + filename))
    if len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def open_cygnss_csvs():
    df_list = []
    for filename in os.listdir('cygnss_data'):
        df_list.append(pd.read_csv('cygnss_data/' + filename))
    if len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def open_anomalies_csvs():
    df_list = []
    for filename in os.listdir('mss_ano_df'):
        df_list.append(pd.read_csv('mss_ano_df/' + filename))
    if len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def prep_cygnss(cygnss_df):
    np.warnings.filterwarnings('ignore')
    for key in cygnss_df:
        cygnss_df = cygnss_df[cygnss_df[key] != -9999.0]
    cygnss_df = cygnss_df[cygnss_df.quality_flags % 2 == 0]
    cygnss_df.dropna(inplace=True)
    cygnss_df['nbrcs_log'] = 10 * np.log10(cygnss_df.ddm_nbrcs.to_numpy())
    cygnss_df.drop('ddm_nbrcs', inplace=True, axis=1)
    return reduce_area_of_df(cygnss_df)


# SET AREA, Function Is called when extracting CYGNSS, OSKAR and ERA5
def reduce_area_of_df(df):
    df = df[df.sp_lat <= 40]
    df = df[df.sp_lat >= -40]

    df_northern = df[df.sp_lat >= 20]
    df_southern = df[df.sp_lat <= -20]
    df = pd.concat([df_northern, df_southern])
    df = df[df.sp_lon >= 220]
    return df[df.sp_lon <= 260]


def get_era_5(filename):
    ds = xr.open_dataset(filename)
    era_5_df = ds.to_dataframe()
    index_long = era_5_df.index.levels[0]
    index_lat = era_5_df.index.levels[1]
    index_time = era_5_df.index.levels[2]

    start_time = pd.Timestamp(1992, 10, 5)
    index_time_fixed = []
    for tid in index_time:
        hours = tid - start_time
        index_time_fixed.append(hours.days * 24 + hours.seconds / 3600)

    long, lat, time = np.meshgrid(index_long, index_lat, index_time_fixed, indexing='ij')

    long = long.flatten()
    lat = lat.flatten()
    time = time.flatten()

    return pd.DataFrame({'sp_lon': long + 360, 'sp_lat': lat, 'hours_since_ref': time, 'u10': era_5_df["u10"].to_numpy()
                            , 'v10': era_5_df["v10"].to_numpy()})

read = False
if read:
    cygnss_df = open_cygnss_csvs()
    # If you want to change area
    cygnss_df = reduce_area_of_df(cygnss_df)
else:
    cygnss_df = fetch_cygnss(2021, 10, 18, 2021, 10, 18)
cygnss_df