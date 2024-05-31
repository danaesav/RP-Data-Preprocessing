import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

from DataProcessor import DataProcessor, select_scattered_points
from gen_adj_mx import analyze_adj_mx
from generator import generate_h5_files, generate_adj_mxs, generate_plots
from plotting import plot_scalability, plot_complexity, get_wandb_df, plot_performance

# metrla_box_coordinates = [34.174317, -118.409044, -118.345701, 34.131097]
metrla_box_coordinates_bigger = [34.18227, -118.511454, -118.345701, 34.131097]
metrla_box_coordinates_2 = [34.097491, -118.265575, -118.207500, 34.048523]
metrla_box_coordinates = [34.188469, -118.509482, -118.439572, 34.132489]

pemsbay_box_coordinates = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_bigger = [37.393346, -121.952686, -121.873484, 37.329885]
pemsbay_box_coordinates_2 = [37.421656, -122.085214, -122.038178, 37.378017]

# [road_distance_small, sensor_ids_file, dataset_file, coordinates, dataset_name, coordinates_bigger, distances_filename, road_distance_large]
metrla = ["METRLA", "metr_ids.txt", "metr-la", metrla_box_coordinates, "METR-LA", metrla_box_coordinates_bigger,
          "distances_la_2012.csv"]
metrla2 = ["METRLA", "metr_ids.txt", "metr-la", metrla_box_coordinates_2, "METR-LA", metrla_box_coordinates_bigger,
           "distances_la_2012.csv"]
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

def update_wandb():
    wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    # for run in runs:
    #     run.summary["AVG Training Time/nodes"] = run.summary['AVG Training time secs/epoch']/run.config['Nodes']
    #     run.summary.update(
    #         {"Average GPU % Usage": np.mean(run.history(stream="events").loc[:, "system.gpu.process.0.gpu"])})
    #     run.update()
    for run in runs:
        data1 = None
        if run.config["Type"] == "original":
            data1 = pd.read_hdf("Datasets/" + run.config["Dataset"] + "/" + run.config["Dataset"].lower() + ".h5")
        else:
            data1 = pd.read_hdf("../D2STGNN-github/datasets/raw_data/" + run.config["Dataset"] + "/" + run.name + ".h5")
        zero_count1 = (data1 == 0).sum().sum()
        print("Percentage of missing values in ", run.name, ": ", zero_count1 / (data1.shape[0] * data1.shape[1]) * 100)
        run.summary.update(
            {"Average GPU % Usage": np.mean(run.history(stream="events").loc[:, "system.gpu.process.0.gpu"])})
        run.update()
        run.summary["AVG Training Time/nodes"] = run.summary['AVG Training time secs/epoch'] / run.config['Nodes']
        run.config["Missing values %"] = zero_count1 / (data1.shape[0] * data1.shape[1]) * 100
        run.update()


def scalability(scalability_data):
    # plot_scalability("Missing values %", scalability_data, "missing_values")
    # plot_scalability("Mean Absolute Error (Horizons Average)", scalability_data, "mae")
    # plot_scalability("Root Mean Squared Error (Horizons Average)", scalability_data, "rmse")
    # plot_scalability("Average Training Time (secs/epoch)", scalability_data, "time")
    # plot_scalability("Average Training Time/node", scalability_data, "time-per-node")
    # plot_scalability("Average GPU % Used", scalability_data, "gpu")
    # plot_scalability("Average Node Neighbors", scalability_data, "neighbors")
    # plot_scalability("Average Neighbors Ratio", scalability_data, "neigh_ratio")
    plot_scalability("Edges", scalability_data, "edges")


def complexity(complexity_data, dataset):
    # plot_complexity("Mean Absolute Error (Horizons Average)", complexity_data, dataset, "mae")
    # plot_complexity("Root Mean Squared Error (Horizons Average)", complexity_data, dataset, "rmse")
    # plot_complexity("Missing values %", complexity_data, dataset, "missing_values")
    # plot_complexity("Average Training Time (secs/epoch)", complexity_data, dataset, "time")
    # plot_complexity("Average Training Time/node", complexity_data, dataset, "time-per-node")
    # plot_complexity("Average GPU % Used", complexity_data, dataset, "gpu")
    # plot_complexity("Average Node Neighbors", complexity_data, dataset, "neighbors")
    # plot_complexity("Average Neighbors Ratio", complexity_data, dataset, "neigh_ratio")
    plot_complexity("Edges", complexity_data, dataset, "edges")


def stand_dev(dataset, typ, dataset_name, param):
    filter_type = dataset[dataset['Type'] == typ]
    filtered = filter_type[filter_type['Dataset'] == dataset_name]
    print(dataset_name,"-", typ, filtered[param].std())



def performance(runs, dataset):
    plot_performance("Test MAE (AVG)", runs, dataset, "mae", "Test MAE (Horizon's Average)")
    plot_performance("Test RMSE (AVG)", runs, dataset, "rmse", "Test RMSE (Horizon's Average)")


if __name__ == '__main__':
    # update_wandb()
    wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    # analyze_adj_mx()

    # performance(runs, "METR-LA")
    # performance(runs, "PEMS-BAY")

    # processor = DataProcessor(data_option, sensor_locations_file)
    # in_box, in_comp_box, out_of_box = processor.process_data()
    # scattered_points = select_scattered_points(in_box, len(in_box))
    # processor.plot_data("comparison", in_box, in_box, out_of_box)

    data = get_wandb_df(runs)
    # stand_dev(data, "small", "METR-LA", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data, "large", "METR-LA", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data, "small", "PEMS-BAY", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data, "large", "PEMS-BAY", "Mean Absolute Error (Horizons Average)")

    scalability(data)

    complexity(data, "METR-LA")
    complexity(data, "PEMS-BAY")