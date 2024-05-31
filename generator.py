import os
import pickle

import cv2
import pandas as pd

from DataProcessor import DataProcessor
from gen_adj_mx import get_adjacency_matrix

sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 1 / 3, 0.25]
suffixes = ["100", "075", "050", "030", "025"]


def generate_scattered_points(processor, within_box, in_comparison_box, outside_of_box, h5_filename, data_option):
    for size, suffix in zip(sizes, suffixes):
        scattered_points_small = processor.save_filtered_data(within_box, len(within_box) * size,
                                                              data_option[4] + "/" + f"{h5_filename}-small-{suffix}")
        processor.plot_data(f"{h5_filename}-small-{suffix}", scattered_points_small, within_box, outside_of_box)
        scattered_points_large = processor.save_filtered_data(in_comparison_box, len(in_comparison_box) * size,
                                                              data_option[4] + "/" + f"{h5_filename}-large-{suffix}")
        processor.plot_data(f"{h5_filename}-large-{suffix}", scattered_points_large, in_comparison_box, outside_of_box)


def save_adj_mx(filename, option):
    with open("ids/" + filename + ".txt") as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv("sensor_graph/" + option[6], dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
    # Save to pickle file.
    if not os.path.exists("../D2STGNN-github/datasets/sensor_graph/adj_mxs/" + option[4]):
        os.makedirs("../D2STGNN-github/datasets/sensor_graph/adj_mxs/" + option[4])
    with open("../D2STGNN-github/datasets/sensor_graph/adj_mxs/" + filename + ".pkl", 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)


def generate_adj_mxs(option):
    for size, suffix in zip(sizes, suffixes):
        save_adj_mx(option[4] + "/" + f"{option[2]}-small-{suffix}", option)
        save_adj_mx(option[4] + "/" + f"{option[2]}-large-{suffix}", option)


def generate_h5_files(option):
    processor = DataProcessor(option, sensor_locations_file)
    within_box, in_comparison_box, outside_of_box = processor.process_data()
    for size, suffix in zip(sizes, suffixes):
        processor.save_filtered_data(within_box, len(within_box) * size, f"{option[2]}-small-{suffix}")
        processor.save_filtered_data(in_comparison_box, len(in_comparison_box) * size, f"{option[2]}-large-{suffix}")


def concat_images(option):
    large_images = []
    small_images = []
    for size, suffix in zip(sizes, suffixes):
        large_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-large-{suffix}.png"))
        small_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-small-{suffix}.png"))
    large_image = cv2.vconcat(large_images)
    small_image = cv2.vconcat(small_images)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-large.png", large_image)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-small.png", small_image)
    # show the output image
    # cv2.imshow('sea_image.jpg', im_v)
