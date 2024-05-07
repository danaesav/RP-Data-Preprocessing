import pickle

import pandas as pd

from DataProcessor import DataProcessor
from gen_adj_mx import get_adjacency_matrix

metrla_box_coordinates = [34.174317, -118.409044, -118.345701, 34.131097]
metrla_box_coordinates_bigger = [34.18227, -118.511454, -118.345701, 34.131097]
pemsbay_box_coordinates = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_bigger = [37.393346, -121.952686, -121.873484, 37.329885]
metrla = ["METRLA", "metr_ids.txt", "metr-la", metrla_box_coordinates, "METR-LA", metrla_box_coordinates_bigger,
          "distances_la_2012.csv"]
pemsbay = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay", pemsbay_box_coordinates, "PEMS-BAY",
           pemsbay_box_coordinates_bigger, "distances_bay_2017.csv"]
sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 1 / 3, 0.25]
suffixes = ["100", "075", "050", "030", "025"]

data_option = pemsbay
h5_filename = data_option[2]
distances_filename = data_option[6]


def generate_scattered_points():
    for size, suffix in zip(sizes, suffixes):
        scattered_points_small = processor.save_filtered_data(within_box, len(within_box) * size,
                                                              data_option[4] + "/" + f"{h5_filename}-small-{suffix}")
        processor.plot_data(f"{h5_filename}-small-{suffix}", scattered_points_small, within_box, outside_of_box)
        scattered_points_large = processor.save_filtered_data(in_comparison_box, len(in_comparison_box) * size,
                                                              data_option[4] + "/" + f"{h5_filename}-large-{suffix}")
        processor.plot_data(f"{h5_filename}-large-{suffix}", scattered_points_large, in_comparison_box, outside_of_box)


def save_adj_mx(filename):
    with open("ids/" + filename + ".txt") as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv("sensor_graph/" + distances_filename, dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
    # Save to pickle file.
    with open("adj_mxs/" + filename + ".pkl", 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)


def generate_adj_mxs():
    for size, suffix in zip(sizes, suffixes):
        save_adj_mx(data_option[4] + "/" + f"{h5_filename}-small-{suffix}")
        # save_adj_mx(data_option[4] + "/" + f"{h5_filename}-large-{suffix}")


if __name__ == '__main__':
    processor = DataProcessor(data_option, sensor_locations_file)
    within_box, in_comparison_box, outside_of_box = processor.process_data()
    # generate_scattered_points()
    generate_adj_mxs()
