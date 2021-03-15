import pandas
import numpy as np
import random
import json


def get_time_series(df, time_steps):
    '''
    Creates a time series with a value for each time step in the form
    [0,0,0,1,1,1,1,-1,-1,-1]
    '0' means no event, '1' is open, '-1' is shut.
    The input data frame is assumed to contain data for single well,
    single zone, single realisation.
    '''
    if df.shape[0] == 0:
        return [0] * len(time_steps)

    result = []
    d = df.sort_values(by=['DATE'])

    is_open = 0
    c = 0
    t0 = d['DATE'].iat[0]
    for t in time_steps:
        if t == t0:
            v = d['OP/SH'].iat[c]
            if v == 'OPEN':
                is_open = 1
            elif v == 'SHUT':
                is_open = -1
            c += 1
            if c < d.shape[0]:
                t0 = d['DATE'].iat[c]
        result.append(is_open)
    return result


def extract_completions(df, layer_to_zone, time_steps, realisations):
    '''
    Extracts completions into a dictionary on the form
    {
        ("zone1", 0): [ [ time_sequence_realization_1 ],
                   [ time_sequence_realization_2 ],
                    ...
        ("zone2", 1): ---
    }
    The key is the tuple of zone and layer (multiple layers can be mapped to one zone)
    Full matrix - every time step and realisation.
    '''
    completions = {}
    for layer, zone_name in layer_to_zone.items():
        # TODO: assuming K1 == K2...
        data = df.loc[df['K1'] == layer]
        layer_data = []
        for rname, realdata in data.groupby('REAL'):
            layer_data.append(get_time_series(realdata, time_steps))
        if len(layer_data) > 0:
            completions[(zone_name, layer)] = layer_data
    return completions


def format_time_series(series):
    '''
    The function compresses the open/shut state from a value for every time step:
      ([0, 0, 0, 0.25, 0.25, 1.0, 0], # open state
       [0, 0, 0, 0,    0,    0,   1.0] # shut state
       )
    into a more compact form:
      ([3, 5, 6],        # time steps when the state changes
       [0.25, 1.0, 0.0], # open state
       [0,     0,  1.0]  # shut state
       )
    '''
    time_steps = []
    values = []
    shut_values = []

    time_series, shut_series = series
    n = len(time_series)
    v0 = time_series[0]
    s0 = shut_series[0]
    if v0 > 0. or s0 > 0.:
        time_steps.append(0)
        values.append(v0)
        shut_values.append(s0)

    for i in range(1, n):
        v = time_series[i]
        s = shut_series[i]
        if v != v0 or s != s0:
            time_steps.append(i)
            values.append(v)
            shut_values.append(s)
            v0 = v
            s0 = s

    if len(time_steps) == 0:
        return None
    return (time_steps, values, shut_values)


def extract_kh(df, layer_to_zone):
    data_by_zone = {}
    for layer, zone_name in layer_to_zone.items():
        # TODO: assuming K1 == K2...
        d = df.loc[df['K1'] == layer]
        if d.shape[0] > 0:
            data = d['KH'].to_numpy()
            if zone_name in data_by_zone:
                data = np.concatenate((data_by_zone[zone_name], data))
            data_by_zone[zone_name] = data
    return data_by_zone


def get_open_shut_fractions(completions, realisation_count):
    open_shut_frac = {}
    for zone_layer, values in completions.items():
        # TODO: Multiple layers per zone is not handled properly
        # Only one of the layers will be captured

        zone_name, layer = zone_layer

        # get rid of the negative "shut"-values
        open_count = np.maximum(np.asarray(values), 0)
        # sum over realisations
        open_count_reduced = open_count.sum(axis=0) / float(realisation_count)

        shut_count = np.maximum(np.asarray(values)*(-1.), 0)
        # sum over realisations
        shut_count_reduced = shut_count.sum(axis=0) / float(realisation_count)

        # fraction of open/shut realisations
        open_shut_frac[zone_name] = (
            np.asarray(open_count_reduced, dtype=np.float64),
            np.asarray(shut_count_reduced, dtype=np.float64)
        )

    return open_shut_frac


def extract_well_completions(df, layer_to_zone, time_steps, realisations):

    completions = extract_completions(
        df, layer_to_zone, time_steps, realisations)

    open_shut_frac = get_open_shut_fractions(completions, len(realisations))

    # retrieve the KH values for all realisations in a dictionary
    kh_all_zones = extract_kh(df, layer_to_zone)

    result = {}
    for zone_name, series in open_shut_frac.items():
        formatted_time_series = format_time_series(series)
        if formatted_time_series is not None:
            r = {}
            r['t'] = formatted_time_series[0]
            r['open'] = formatted_time_series[1]
            r['shut'] = formatted_time_series[2]
            kh = kh_all_zones[zone_name]
            r['khMean'] = kh.mean()
            r['khMin'] = np.amin(kh)
            r['khMax'] = np.amax(kh)
            result[zone_name] = r
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
