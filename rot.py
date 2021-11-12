# Method of Andreas
from datetime import date


def collect_tracks():
    track_list = []
    for satellite in range(1, 2): #Change from 1 , 9
        print("satellite: "+str(satellite))
        test_data_url, test_clickable_url = generate_url(2021, 10, 31, satellite)
        try:
            dataset = open_url(test_data_url, output_grid=False)
        except:
            dataset = None
        if dataset is None:
            print("Something went wrong with dataset collection")
            pass
        else:
            for ddm in range(1):
                track_id = np.array(dataset.track_id[:, ddm]).flatten()
                track_indices = np.append(np.append(-1, np.where(np.diff(track_id) > 0)), len(track_id))

                sp_lat = np.array(dataset.sp_lat[:, ddm]).flatten().tolist()
                sp_lon = np.array(dataset.sp_lon[:, ddm])

                # We need to convert longitude to [-180, 180] (ERA5 Winds)
                a, b = (np.where(sp_lon > 180))
                sp_lon[a] -= 360

                nbrcs = np.array(dataset.ddm_nbrcs[:, ddm]).flatten().tolist()
                ddm_timestamp_utc = np.array(dataset['ddm_timestamp_utc'][:]).tolist()
                quality_flags = np.array(dataset['quality_flags'][:, ddm]).flatten().tolist()

                for track in range(len(track_indices)-1):
                    start = track_indices[track]+1
                    end = track_indices[track+1]

                    lats = sp_lat[start: end]
                    lons = sp_lon[start: end].flatten().tolist()
                    track_nbrcs = nbrcs[start: end]
                    track_ddm_timestamp_utc = ddm_timestamp_utc[start: end]
                    track_quality_flags = quality_flags[start: end]
                    track_list.append([lats, lons, track_nbrcs, track_ddm_timestamp_utc, track_quality_flags])

    return track_list

total_df = total_df.reset_index(drop =True)
prev = total_df['track_id'] [0]
indices_change_track_id = []
for index, row in total_df.iterrows():
    if prev != row['track_id']:
        indices_change_track_id.append(index)
        prev = row['track_id']

        '''
        d0 = date(1992, 10, 5)
        d1 = date(2008, 9, 26)
        delta = d1 - d0
        dataset = open_url('https://podaac-opendap.jpl.nasa.gov/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel2011.nc.gz', output_grid=False)
        oskar_df = pd.DataFrame()
        print(len(np.array(dataset.latitude)))
        print(len(np.array(dataset.longitude)))
        oskar_df['lat'] = np.array(dataset.latitude)
        oskar_df['long'] = np.array(dataset.longitude)
        oskar_df['u'] = np.array(dataset.u)
        oskar_df['v'] = np.array(dataset.v)
        oskar_df
        remote_data = xr.open_dataset('https://podaac-opendap.jpl.nasa.gov/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel2011.nc.gz',decode_times=False)
        print(len(np.array(remote_data['latitude'])))
        print(len(np.array(remote_data['longitude'])))
        '''
def get_closest_oskar_url(year, month, day):
    d0 = date(1992, 10, 5)
    d1 = date(year, month, day)
    delta = d1 - d0
    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel' + str(delta.days) + '.nc.gz'
    return base_url
