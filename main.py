import os
import pickle

import numpy as np
import pandas as pd
import wandb
import cv2

from DataProcessor import DataProcessor
from gen_adj_mx import get_adjacency_matrix

metrla_box_coordinates = [34.174317, -118.409044, -118.345701, 34.131097]
metrla_box_coordinates_bigger = [34.18227, -118.511454, -118.345701, 34.131097]
metrla_box_coordinates_2 = [34.097491, -118.265575, -118.207500, 34.048523]

pemsbay_box_coordinates = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_bigger = [37.393346, -121.952686, -121.873484, 37.329885]
pemsbay_box_coordinates_2 = [37.285771, -121.983673, -121.915783, 37.240070]

# [road_distance_small, sensor_ids_file, dataset_file, coordinates, dataset_name, coordinates_bigger, distances_filename, road_distance_large]
metrla = [7.5, "metr_ids.txt", "metr-la", metrla_box_coordinates, "METR-LA", metrla_box_coordinates_bigger,
          "distances_la_2012.csv", 15.4]
metrla2 = [7.5, "metr_ids.txt", "metr-la", metrla_box_coordinates_2, "METR-LA", metrla_box_coordinates_bigger,
          "distances_la_2012.csv", 15.4]
pemsbay = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay", pemsbay_box_coordinates, "PEMS-BAY",
           pemsbay_box_coordinates_bigger, "distances_bay_2017.csv"]
pemsbay2 = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay", pemsbay_box_coordinates_2, "PEMS-BAY",
            pemsbay_box_coordinates_bigger, "distances_bay_2017.csv"]
sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 1 / 3, 0.25]
suffixes = ["100", "075", "050", "030", "025"]

data_option = metrla
h5_filename = data_option[2]
distances_filename = data_option[6]


def generate_scattered_points():
    x = 3
    # for size, suffix in zip(sizes, suffixes):
    #     scattered_points_small = processor.save_filtered_data(within_box, len(within_box) * size,
    #                                                           data_option[4] + "/" + f"{h5_filename}-small-{suffix}")
    #     processor.plot_data(f"{h5_filename}-small-{suffix}", scattered_points_small, within_box, outside_of_box)
    #     scattered_points_large = processor.save_filtered_data(in_comparison_box, len(in_comparison_box) * size,
    #                                                           data_option[4] + "/" + f"{h5_filename}-large-{suffix}")
    #     processor.plot_data(f"{h5_filename}-large-{suffix}", scattered_points_large, in_comparison_box, outside_of_box)


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


def save_filtered(process, small_box, large_box, option):
    for size, suffix in zip(sizes, suffixes):
        process.save_filtered_data(small_box, len(small_box) * size, f"{option[2]}-small-{suffix}")
        process.save_filtered_data(large_box, len(large_box) * size, f"{option[2]}-large-{suffix}")


# def generate_h5_files(option):
#     processor = DataProcessor(option, sensor_locations_file)
#     within_box, in_comparison_box, outside_of_box = processor.process_data()
#     save_filtered(processor, within_box, in_comparison_box, option)

def update_wandb():
    wandb.login(key='f08a2911f23c8c1152cbf05edfc79d1e1fcc8fc1')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN")
    for run in runs:
        print(run.name)
        t = run.name.split('-')[2]
        size = run.name.split('-')[-1]
        run.summary.update(
            {"Average GPU % Usage": np.mean(run.history(stream="events").loc[:, "system.gpu.process.0.gpu"])})
        # run.summary.update({"Type": t})
        # run.summary.update({"Size": size})
        run.config["Type"] = t
        run.config["Size"] = size
        run.update()


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


if __name__ == '__main__':
    processor = DataProcessor(metrla, sensor_locations_file)
    within_box, in_comparison_box, outside_of_box = processor.process_data()
    processor = DataProcessor(metrla2, sensor_locations_file)
    within_box2, in_comparison_box2, outside_of_box2 = processor.process_data()
    processor.save_filtered_data(within_box2, len(within_box2), f"metr-la-comparison-100")
    save_adj_mx(metrla[4] + "/" + f"{metrla[2]}-comparison-100", metrla)

    processor2 = DataProcessor(pemsbay, sensor_locations_file)
    P_within_box, P_in_comparison_box, P_outside_of_box = processor2.process_data()
    processor2 = DataProcessor(pemsbay2, sensor_locations_file)
    P_within_box2, P_in_comparison_box2, P_outside_of_box2 = processor2.process_data()
    processor2.save_filtered_data(P_within_box2, len(P_within_box2), f"pems-bay-comparison-100")
    save_adj_mx(pemsbay[4] + "/" + f"{pemsbay[2]}-comparison-100", pemsbay)

    # processor.plot_data(f"{h5_filename}-areas-comparison", within_box, within_box2, in_comparison_box, outside_of_box)

    # generate_adj_mxs(metrla)
    # generate_h5_files(metrla)
    #
    # generate_adj_mxs(pemsbay)
    # generate_h5_files(pemsbay)
