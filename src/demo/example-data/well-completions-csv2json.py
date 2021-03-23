import pandas
import numpy as np
import random
import json
import math


def get_time_series(df, time_steps):
    '''
    Input:
      Pandas data frame containing data for a single well, single zone, single realisation.
      Sorted list of all time steps in the dataset.

    Creates time series for open/shut state and KH.
    The open/shut info is formatted like this:
    [0,0,0,1,1,1,1,-1,-1,-1]
    '0' means completion not existing, '1' is open, '-1' is shut.
    KH:
    [nan, nan, nan, value1, value1, value1, value1, value2, value2, value2]
    '''
    result = []
    result_kh = []
    d = df.sort_values(by=['DATE'])

    is_open = 0
    kh = math.nan
    c = 0
    t0 = d['DATE'].iat[0]
    for t in time_steps:
        if t == t0:
            v = d['OP/SH'].iat[c]
            kh = d['KH'].iat[c]
            if v == 'OPEN':
                is_open = 1
            elif v == 'SHUT':
                is_open = -1
            c += 1
            if c < d.shape[0]:
                t0 = d['DATE'].iat[c]
        result.append(is_open)
        result_kh.append(kh)
    return (result, result_kh)


def get_completions_by_zone(df, layer_to_zone, time_steps, realisations):
    '''
    Extracts completions into a dictionary of 2D arrays on the form
    {
        ("zone1", 0): {
            'opsh': 2D array of open/shut values encoded as 1 or -1. 0 for missing.
            'kh': 2D array of kh values. Nan for missing.
        ("zone2", 1): ---
    }
    2D arrays have dimension [time_steps x realisations].
    The dictionary key is the tuple of zone and layer (multiple layers can be mapped to one zone).
    '''
    completions = {}
    for layer, zone_name in layer_to_zone.items():
        # TODO: assuming K1 == K2...
        data = df.loc[df['K1'] == layer]

        layer_data = []
        kh_data = []
        for rname, realdata in data.groupby('REAL'):
            comp, kh = get_time_series(realdata, time_steps)
            layer_data.append(comp)
            kh_data.append(kh)

        if len(layer_data) > 0:
            d = {}
            d['opsh'] = layer_data
            d['kh'] = kh_data
            completions[(zone_name, layer)] = d
    return completions


def compress_time_series(series):
    '''
    Input:
      Dictionary of time series. Must contain series for 'open' and 'shut'.

    The function uses the open/shut state to compress the time series.
    The inital time steps are skipped, if the completion does not exist in any realisation.
    Then only the time steps where open/shut state changes is captured.

    Example:
      ([0, 0, 0, 0.25, 0.25, 1.0, 0], # open state
       [0, 0, 0, 0,    0,    0,   1.0] # shut state
       )
    into a more compact form:
      ([3, 5, 6],        # time steps when the state changes
       [0.25, 1.0, 0.0], # open state
       [0,     0,  1.0]  # shut state
       )

    Any additional time series (KH) is transfered using the same time steps sampling.
    '''
    time_steps = []

    result = {}
    result['t'] = []
    for key in series.keys():
        result[key] = []

    open_series = series['open']
    shut_series = series['shut']

    is_open = 0
    is_shut = 0
    n = len(open_series)
    for i in range(0, n):
        o = open_series[i]
        s = shut_series[i]
        if (i == 0 and (o > 0. or s > 0.)) or o != is_open or s != is_shut:
            time_steps.append(i)
            for key in series.keys():
                result[key].append(series[key][i])
            is_open = open_series[i]
            is_shut = shut_series[i]

    if len(time_steps) == 0:
        return None
    result['t'] = time_steps
    return result


def get_kh_stats_series(values):
    '''
    Takes 2D array of KH as input (time x realisation).
    Creates statistics: mean, max and min as function of time.
    '''
    kh_arr2d = np.asarray(values)

# this works, but gives runtime warnings
#    kh_avg = np.nanmean(kh_arr2d, axis=0)
#    kh_min = np.nanmin(kh_arr2d, axis=0)
#    kh_max = np.nanmax(kh_arr2d, axis=0)

    # process timesteps one by one. Check for all nans.
    kh_avg = []
    kh_min = []
    kh_max = []
    for i in range(kh_arr2d.shape[1]):
        if np.count_nonzero(~np.isnan(kh_arr2d[:, i])) > 0:
            kh_avg.append(np.nanmean(kh_arr2d[:, i]))
            kh_min.append(np.nanmin(kh_arr2d[:, i]))
            kh_max.append(np.nanmax(kh_arr2d[:, i]))
        else:
            kh_avg.append(math.nan)
            kh_min.append(math.nan)
            kh_max.append(math.nan)

    return {'khMean': kh_avg, 'khMin': kh_min, 'khMax': kh_max}


def get_open_shut_fractions(values, realisation_count):
    '''
    Takes 2D array of open/shut/missing as input, and total number of realisations.
    Calculates the fraction of open and shut for each time step.
    '''
    # get rid of the negative "shut"-values
    open_count = np.maximum(np.asarray(values), 0)
    # sum over realisations
    open_count_reduced = open_count.sum(axis=0) / float(realisation_count)

    shut_count = np.maximum(np.asarray(values)*(-1.), 0)
    # sum over realisations
    shut_count_reduced = shut_count.sum(axis=0) / float(realisation_count)

    # fraction of open/shut realisations
    return {
        'open': np.asarray(open_count_reduced, dtype=np.float64),
        'shut': np.asarray(shut_count_reduced, dtype=np.float64)
    }


def extract_well_completions(df, layer_to_zone, time_steps, realisations):
    '''
    Input:
      Pandas data frame for one well.
      Map from layer index to zone name.
      All time steps (sorted).
      All realisations.

    Returns completion time-series for the well aggreated over realisations.
    '''
    completions = get_completions_by_zone(
        df, layer_to_zone, time_steps, realisations)

    result = {}
    for zone_name_layer, comps in completions.items():

        # TODO: not combining result for different layers merged into the same zone yet
        zone_name, layer = zone_name_layer

        series = get_open_shut_fractions(comps['opsh'], len(realisations))
        series.update(get_kh_stats_series(comps['kh']))

        formatted_time_series = compress_time_series(series)
        if formatted_time_series is not None:
            result[zone_name] = formatted_time_series
    return result


def extract_wells(df, layer_to_zone, time_steps, realisations):
    well_list = []
    for well_name, well_group in df.groupby('WELL'):
        well = {}
        well['name'] = well_name
        well['completions'] = extract_well_completions(well_group, layer_to_zone,
                                                       time_steps, realisations)
        well_list.append(well)
    return well_list


def random_color_str():
    r = random.randint(8, 15)
    g = random.randint(8, 15)
    b = random.randint(8, 15)
    s = hex((r << 8) + (g << 4) + b)
    return "#" + s[-3:]


def extract_stratigraphy(zone_names):
    result = []
    for zone_name in zone_names:
        zdict = {}
        zdict['name'] = zone_name
        zdict['color'] = random_color_str()
        result.append(zdict)
    return result


def create_well_completion_dict(filename):

    time_steps = sorted(pandas.unique(df['DATE']))
    realisations = np.asarray(
        sorted(pandas.unique(df['REAL'])), dtype=np.int32)

    layers = np.sort(pandas.unique(df['K1']))

    # construct a map from layer to zone name
    # NOTE: multiple layers mapped to the same zone does not work.
    layer_to_zone = {}
    zone_names = []
    for layer in layers:
        zone_name = 'zone' + str(layer)
        layer_to_zone[layer] = zone_name
        zone_names.append(zone_name)

    result = {}
    result['stratigraphy'] = extract_stratigraphy(zone_names)
    result['timeSteps'] = time_steps
    result['wells'] = extract_wells(
        df, layer_to_zone, time_steps, realisations)
    return result


def add_well_attributes(wells):
    for well in wells:
        # TODO: Should make some more interesting well attributes
        well['type'] = 'Producer'
        well['region'] = 'Region1'


if __name__ == '__main__':
    # fixed seed to avoid different colors between runs
    random.seed(1234)
    filename = 'compdat.csv'
    df = pandas.read_csv(filename)
    result = create_well_completion_dict(df)
    add_well_attributes(result['wells'])
    # json_str = json.dumps(result)

    # more human friendly output:
    json_str = json.dumps(result, indent=2)
    print(json_str)
