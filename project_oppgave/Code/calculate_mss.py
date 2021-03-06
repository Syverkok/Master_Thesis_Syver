from functions import *
from tqdm import tqdm
oskar_df = fetch_all_oskar_files()
read = True
bias_df = pd.read_csv('../bias_model.csv')
interp_bias = LinearNDInterpolator(list(zip(bias_df['inc_angle'], bias_df['delta'])), bias_df['bias'])
lats_north = np.linspace(-40, -20, 11)
#lats_north = np.linspace(20, 40, 11)
longs = np.linspace(220, 260, 11)

for i in tqdm(range(len(lats_north) - 1)):
    lat_start, lat_end = lats_north[i], lats_north[i + 1]
    for j in range(len(longs) - 1):
        long_start, long_end = longs[j], longs[j + 1]

        sub_cyg = open_cygnss_csvs(lat_start, lat_end, long_start, long_end)

        sub_era5 = get_era_5('../era_5_wind/october_november_18_-40_40_220_260.nc',
                             lat_start - 0.26, lat_end + 0.26, long_start - 0.26, long_end + 0.26)

        sub_era5 = sub_era5[sub_era5.hours_since_ref >= 254112.0]
        sub_era5 = reduce_area_of_sub_df(sub_era5, lat_start, lat_end, long_start, long_end)

        sub_osk = reduce_area_of_sub_df(oskar_df, lat_start - 0.34, lat_end + 0.34, long_start - 0.34, long_end + 0.34)

        if (not sub_cyg.empty) and (not sub_era5.empty) and (not sub_osk.empty):
            print("nice")
            mss_ano_df = calculate_mss_anomaly_df(sub_cyg, sub_era5, sub_osk, interp_bias)
            # REMEMBER TO CHANGE FOLDER
            mss_ano_df.to_csv("../test_mss/" + str(lat_start) + str(lat_end) + str(long_start) + str(long_end) +
                              ".csv", index=False)

        else:
            print(sub_cyg.empty)
            print(sub_era5.empty)
            print(sub_osk.empty)