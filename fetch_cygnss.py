from pydap.client import open_url
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


pd.set_option('display.max_columns', None)


def generate_url(year, month, day, satellite_number):

    day_of_year = datetime(year, month, day).timetuple().tm_yday
    # print('day_of_year used for link generation: ', day_of_year)
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url
    clickable_url = base_url + specific_url + '.html'

    return data_url + '?sp_lat,sp_lon,ddm_timestamp_utc,ddm_snr,gps_tx_power_db_w,gps_ant_gain_db_i,rx_to_sp_range,' \
                      'tx_to_sp_range,sp_rx_gain', clickable_url


def collect_dataset(url):
    dataset = open_url(url, output_grid=False)
    print(dataset)
    df = pd.DataFrame()

    for ddm in range(4):  # Remember to change back to 4

        ddm_df = pd.DataFrame()

        print("ddm: " + str(ddm))
        sp_lat = np.array(dataset.sp_lat[:, ddm])
        sp_lon = np.array(dataset.sp_lon[:, ddm])
        a, b = (np.where(sp_lon > 180))
        sp_lon[a] -= 360

        ddm_timestamp_utc = np.array(dataset.ddm_timestamp_utc[:, ddm])
        ddm_snr = np.array(dataset.ddm_snr[:, ddm])
        gps_tx_power_db_w = np.array(dataset.gps_tx_power_db_w[:, ddm])
        gps_ant_gain_db_i = np.array(dataset.gps_ant_gain_db_i[:, ddm])
        rx_to_sp_range = np.array(dataset.rx_to_sp_range[:, ddm])
        tx_to_sp_range = np.array(dataset.tx_to_sp_range[:, ddm])
        sp_rx_gain = np.array(dataset.sp_rx_gain[:, ddm])

        ddm_df['ddm_channel'] = np.zeros(len(sp_lon))
        ddm_df['sp_lat'] = sp_lat.tolist()
        ddm_df['sp_lon'] = sp_lon.tolist()
        ddm_df = ddm_df.assign(ddm_channel=ddm)

        ddm_df['ddm_timestamp_utc'] = ddm_timestamp_utc.tolist()
        ddm_df['ddm_snr'] = ddm_snr.tolist()
        ddm_df['gps_tx_power_db_w'] = gps_tx_power_db_w.tolist()
        ddm_df['gps_ant_gain_db_i'] = gps_ant_gain_db_i.tolist()
        ddm_df['rx_to_sp_range'] = rx_to_sp_range.tolist()
        ddm_df['tx_to_sp_range'] = tx_to_sp_range.tolist()
        ddm_df['sp_rx_gain'] = sp_rx_gain.tolist()

        for col in ddm_df.columns:
            if col != 'ddm_channel' and col != 'ddm_timestamp_utc':
                ddm_df[col] = ddm_df[col].apply(lambda x: x[0])

        df = df.append(ddm_df, ignore_index=True)

    return df

def calculate_sr(snr, p_r, g_t, g_r, d_ts, d_sr):
    # snr(dB), p_r(dBW), g_t(dBi), g_r(dBi), d_ts(meter), d_sr(meter)
    return snr - (10*np.log10(p_r)) - (10*np.log10(g_t)) - (10*np.log10(g_r)) - (20*np.log10(0.19)) + (20*np.log10(d_ts+d_sr)) + (20*np.log10(4*np.pi))


def plot_smap(df, dot_size=0.5):
    plt.scatter(x=list(df['sp_lon']), y=list(df['sp_lat']), c=list(df['sr']), cmap='Spectral', s=dot_size)
    plt.colorbar()
    plt.title('Surface Reflectivity')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('sr_test.svg')
    plt.show()


def testing():
    start = time.time()
    test_data_url, test_clickable_url = generate_url(2021, 10, 31, 1)
    print('URL generated...')
    print('Collecting data from the server using the generated URL...')
    test_data = open_url(test_data_url, output_grid=False)
    print('Data collected in ' + str(round(time.time()-start, 2)) + ' seconds...\n')

    keys = list(test_data.keys())
    print('Chosen parameters: ', keys)
    # print('Original data shape: ', test_data.shape)

    for k in keys:
        print('Key: ', k)
        key = test_data[k]
        print('Key type: ', key)
        print('Key dimensions: ', key.dimensions)
        print('Key shape: ', key.shape)
        # pprint.pprint(key.attributes)
        print('---------------------------------------------------------------------------------')

    # new_test_data = test_data.functions.geogrid(test_data., 24, 7, 17, 14)

    # print('New data shape: ', new_test_data.shape)


def main():
    test_data_url, test_clickable_url = generate_url(2021, 10, 31, 1)
    print(test_clickable_url)
    print(test_data_url)

    test_df = collect_dataset(test_data_url)

    print('Rows before ddm_snr fill value removal: ', test_df.shape[0])
    test_df = test_df[test_df.ddm_snr != -9999.0]
    print('Rows after ddm_snr fill value removal: ', test_df.shape[0])
    # test_df = test_df[test_df.rx_to_sp_range < 1000000]
    # print('Rows after rx filter apllied: ', test_df.shape[0])

    test_df['sr'] = test_df.apply(lambda row: calculate_sr(row.ddm_snr, row.gps_tx_power_db_w, row.gps_ant_gain_db_i, row.sp_rx_gain, row.tx_to_sp_range, row.rx_to_sp_range), axis=1)

    print(test_df.head())

    print('Max value SR: ', test_df['sr'].max())
    print('Min value SR: ', test_df['sr'].min())

    '''filtered_df = test_df[test_df.sp_lat < 24]
    filtered_df = filtered_df[filtered_df.sp_lat > 17]
    filtered_df = filtered_df[filtered_df.sp_lon < 14]
    filtered_df = filtered_df[filtered_df.sp_lon > 7]'''

    plot_smap(test_df)

if __name__ == '__main__':
    main()
