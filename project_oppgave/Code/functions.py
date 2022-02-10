from pydap.client import open_url
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import timedelta
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from datetime import date
import os


def retrieve_microplastics_gt():
    lats = np.zeros(181)
    count = 90
    for i in range(len(lats)):
        lats[i] = count
        count -= 1
    for filename in os.listdir('../microplast_gt'):
        logfilename = filename[:-4] + '_log'
        try:
            df = pd.read_csv('../microplast_gt/' + filename, header=None)

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


def generate_url(year, month, day, satellite_number):
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url
    clickable_url = base_url + specific_url + '.html'

    return data_url + '?sp_lat,sp_lon,track_id,quality_flags,ddm_timestamp_utc,ddm_nbrcs,fresnel_coeff,sp_inc_angle', clickable_url


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

            # MAYBE REMOVE np.rint() which would give continous hours and better estimations?
            ddm_timestamp_utc = np.rint(ddm_timestamp_utc / 3600) + hours

            ddm_df['sp_lat'] = np.array(dataset.sp_lat[:, ddm]).tolist()
            ddm_df['sp_lon'] = np.array(dataset.sp_lon[:, ddm]).tolist()
            ddm_df['hours_since_ref'] = ddm_timestamp_utc.tolist()
            ddm_df['ddm_nbrcs'] = np.array(dataset.ddm_nbrcs[:, ddm]).tolist()
            ddm_df['fresnel_coeff'] = np.array(dataset.fresnel_coeff[:, ddm]).tolist()
            ddm_df['sp_inc_angle'] = np.array(dataset.sp_inc_angle[:, ddm]).tolist()
            ddm_df['quality_flags'] = np.array(dataset.quality_flags[:, ddm]).tolist()
            '''
            track_id = np.array(dataset.track_id[:, ddm])
            ddm_df['ddm_channel'] = np.zeros(len(sp_lon))
            ddm_df = ddm_df.assign(ddm_channel=ddm)
            ddm_df['track_id'] = track_id.tolist()            
            # ODD quality flagg means bad data, so should be dropped

            '''
            for col in ddm_df.columns:
                if col != 'ddm_channel' and col != 'hours_since_ref':
                    ddm_df[col] = ddm_df[col].apply(lambda x: x[0])
            cygnss_df = cygnss_df.append(ddm_df, ignore_index=True)
    return cygnss_df


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
    for filename in os.listdir('../oskar_data'):
        df_list.append(open_oskar_data_local('../oskar_data/' + filename))
    if len(df_list) == 1:
        return df_list[0]
    else:
        return pd.concat(df_list)


def open_cygnss_csvs(lat_start, lat_end, long_start, long_end):
    df_list = []
    for filename in os.listdir('../cygnss_data_whole_world'):
        df = pd.read_csv('../cygnss_data_whole_world/' + filename)
        df = reduce_area_of_sub_df(df, lat_start, lat_end, long_start, long_end)
        df_list.append(df)
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


def get_era_5(filename, lat_start, lat_end, long_start, long_end):
    ds = xr.open_dataset(filename)
    era_5_df = ds.to_dataframe()
    era_5_df = era_5_df.dropna()
    # era_5_df = reduce_area_of_df_era_5_multi(ds.to_dataframe(), lat_start, lat_end, long_start, long_end)
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

    return pd.DataFrame({'sp_lon': long + 360, 'sp_lat': lat, 'hours_since_ref': time,
                         'u10': era_5_df["u10"].to_numpy(),
                         'v10': era_5_df["v10"].to_numpy()})


def reduce_area_of_df_era_5_multi(df, lat_start, lat_end, long_start, long_end):
    df = df[df.index.get_level_values(0) + 360 > lat_start]
    df = df[df.index.get_level_values(0) + 360 < lat_end]
    df = df[df.index.get_level_values(1) < long_end]
    df = df[df.index.get_level_values(1) > long_start]
    return df


# SET AREA, Function Is called when extracting CYGNSS, OSKAR and ERA5
def reduce_area_of_df(df):
    df = df[df.sp_lat <= 40]
    df = df[df.sp_lat >= -40]

    df_northern = df[df.sp_lat >= 20]
    df_southern = df[df.sp_lat <= -20]
    df = pd.concat([df_northern, df_southern])
    df = df[df.sp_lon >= 220]
    return df[df.sp_lon <= 260]


def reduce_area_of_sub_df(df, lat_start, lat_end, long_start, long_end):
    df = df[df.sp_lat < lat_end]
    df = df[df.sp_lat > lat_start]
    df = df[df.sp_lon > long_start]
    return df[df.sp_lon < long_end]


@np.vectorize
def caluate_mss_towards(u):
    if u <= 3.49:
        return 0.0035 * (u + 0.62)
    else:
        return 0.0035 * (6 * np.log(u) - 3.39)


def calculate_mss_anomaly_df(cygnss_df, era_5_df, oskar_df, interp_bias):
    # Get Wind For CYGNSS
    interp_u10 = LinearNDInterpolator(list(zip(era_5_df['sp_lon'], era_5_df['sp_lat'], era_5_df['hours_since_ref'])),
                                      era_5_df['u10'])
    interp_v10 = LinearNDInterpolator(list(zip(era_5_df['sp_lon'], era_5_df['sp_lat'], era_5_df['hours_since_ref'])),
                                      era_5_df['v10'])

    lons_to_interpolate = cygnss_df["sp_lon"].to_numpy()
    lats_to_interpolate = cygnss_df["sp_lat"].to_numpy()

    times_to_interpolate = cygnss_df["hours_since_ref"].to_numpy()
    u10 = interp_u10(lons_to_interpolate, lats_to_interpolate, times_to_interpolate)
    v10 = interp_v10(lons_to_interpolate, lats_to_interpolate, times_to_interpolate)

    total_wind = np.sqrt(u10 ** 2 + v10 ** 2)

    # GET CURRENT FOR CYGNSS

    interp_u = LinearNDInterpolator(list(zip(oskar_df['sp_lat'], oskar_df['sp_lon'], oskar_df['hours_since_ref'])),
                                    oskar_df['u'])
    interp_v = LinearNDInterpolator(list(zip(oskar_df['sp_lat'], oskar_df['sp_lon'], oskar_df['hours_since_ref'])),
                                    oskar_df['v'])

    times_to_interpolate = cygnss_df["hours_since_ref"].to_numpy()

    u_current = interp_u(lats_to_interpolate, lons_to_interpolate, times_to_interpolate)
    v_current = interp_v(lats_to_interpolate, lons_to_interpolate, times_to_interpolate)

    # MSS ANOMALY CAL
    diff_u = u10 - u_current
    diff_v = v10 - v_current
    delta = np.sqrt(diff_u ** 2 + diff_v ** 2)

    top_frac = 0.059 * delta ** 3 - 0.147 * delta ** 2 + 0.903 ** delta + 1.389
    bot_frac = delta ** 3 + 18.161 * delta ** 2 - 117.602 * delta + 706.9
    mss_from_wind_current = top_frac / bot_frac

    mss_from_wind_towards = caluate_mss_towards(total_wind)

    mss_from_delta_towards = caluate_mss_towards(delta)

    fresnell_sqrd = cygnss_df['fresnel_coeff'].to_numpy() ** 2

    biases = interp_bias(cygnss_df['sp_inc_angle'], delta)

    #mss_from_cygnss = fresnell_sqrd / (10 ** ((cygnss_df['nbrcs_log'] - biases) / 10))

    mss_from_cygnss = fresnell_sqrd / (10 ** ((cygnss_df['nbrcs_log'] - 0) / 10))

    mss_ano_df = pd.DataFrame({'lon': lons_to_interpolate, 'lat': lats_to_interpolate,
                               'mss_cygnss': mss_from_cygnss, 'mss_wind_current': mss_from_wind_current,
                               'mss_towards_wind': mss_from_wind_towards,
                               'mss_towards_delta': mss_from_delta_towards,
                               'nbrcs': cygnss_df['nbrcs_log'], 'wind_u10': u10, 'wind_v10': v10,
                               'current_u': u_current, 'current_v': v_current, 'biases': biases,
                               'fresnel': cygnss_df['fresnel_coeff']})
    return mss_ano_df
