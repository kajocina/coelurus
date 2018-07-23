profiles = data_filter.replicate_data[0].copy()
profiles.set_index('protein_id', inplace=True)
profiles.loc['G0SI40']

for i in range(profiles.shape[1] - 4):
    window = profiles.iloc[:, i:i + 5]
    win_bool = window == 0
    idx_to_flatten = win_bool.apply(lambda x: np.all(x == [True, False, False, False, True]), axis=1
    window.loc[idx_to_flatten, window.columns[1]] = 0
    profiles.iloc[:, i:i + 3] = window
input_data.iloc[:, 1:] = profiles

profiles.loc['G0SI40']