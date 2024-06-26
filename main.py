import os

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ttest_ind

from DataProcessor import DataProcessor, select_scattered_points
from gen_adj_mx import analyze_adj_mx, update_wandb

from plotting import plot_scalability, plot_complexity, get_wandb_df, plot_line_graph, plot_scalability_lineplots, plot_complexity_lineplots

metrla_box_coordinates_level_1_25 = [34.188469, -118.509482, -118.439572, 34.132489]
metrla_box_coordinates_level_2_50 = [34.18227, -118.511454, -118.345701, 34.131097]
metrla_box_coordinates_level_3_75 = [34.18227, -118.511454, -118.272925, 34.131097]
# metrla_box_coordinates_level_4_100 = [34.18227, -118.511454, -118.181559, 34.131097]
metrla_box_coordinates_level_4_100 = [34.18227, -118.511454, -118.211559, 34.131097]
metrla_box_coordinates_level_5_125 = [34.282466, -118.542688, -118.211559, 34.126391]
metrla_box_coordinates_level_6_150 = [34.282466, -118.542688, -118.211559, 34.096391]
metrla_box_coordinates_level_7_175 = [34.282466, -118.542688, -118.149424, 34.082779]
metrla_box_coordinates_comparison = [34.097491, -118.265575, -118.207500, 34.048523]
metrla_coordinates = [metrla_box_coordinates_level_1_25, metrla_box_coordinates_level_2_50,
                      metrla_box_coordinates_level_3_75, metrla_box_coordinates_level_4_100,
                      metrla_box_coordinates_level_5_125, metrla_box_coordinates_level_6_150,
                      metrla_box_coordinates_level_7_175, metrla_box_coordinates_comparison]

pemsbay_box_coordinates_level_1_25 = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_level_2_50 = [37.393346, -121.952686, -121.873484, 37.329885]
pemsbay_box_coordinates_level_3_75 = [37.4041605, -121.970081, -121.876765, 37.321858]
pemsbay_box_coordinates_level_4_100 = [37.414975, -121.987476, -121.880046, 37.313831]
pemsbay_box_coordinates_level_5_150 = [37.413975, -121.990961, -121.830322, 37.303831]
pemsbay_box_coordinates_level_6_200 = [37.429789, -122.020961, -121.830322, 37.290955]
pemsbay_box_coordinates_level_7_250 = [37.429789, -122.020961, -121.830322, 37.200955]
pemsbay_box_coordinates_comparison = [37.421656, -122.085214, -122.038178, 37.378017]
pemsbay_coordinates = [pemsbay_box_coordinates_level_1_25, pemsbay_box_coordinates_level_2_50,
                       pemsbay_box_coordinates_level_3_75, pemsbay_box_coordinates_level_4_100,
                       pemsbay_box_coordinates_level_5_150, pemsbay_box_coordinates_level_6_200,
                       pemsbay_box_coordinates_level_7_250, pemsbay_box_coordinates_comparison]

# [dataset_name, sensor_ids_file, distances_filename, coordinates]
metrla = ["METR-LA", "metr_ids.txt", "distances_la_2012.csv", metrla_coordinates]
pemsbay = ["PEMS-BAY", "pemsbay_ids.txt", "distances_bay_2017.csv", pemsbay_coordinates]
proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
suffixes = ["010", "020", "030", "040", "050", "060", "070", "080", "090", "100"]
less_proportions = [0.25, 0.5, 0.75, 1]
less_suffixes = ["025", "050", "075", "100"]

data_option = metrla
filename_start = data_option[0].lower()


def scalability(scalability_data):
    plot_scalability("Mean Absolute Error (Horizons Average)", scalability_data, "mae")
    plot_scalability("Root Mean Squared Error (Horizons Average)", scalability_data, "rmse")
    plot_scalability("Average Training Time (secs/epoch)", scalability_data, "time")
    plot_scalability("Training Time per Node (seconds)", scalability_data, "time-per-node")
    plot_scalability("Average GPU % Used", scalability_data, "gpu")
    plot_scalability("Average (Node) Degree", scalability_data, "neighbors")
    plot_scalability("Clustering Coefficient", scalability_data, "neigh_ratio")


def complexity(complexity_data):
    plot_complexity("Mean Absolute Error (Horizons Average)", complexity_data, "mae")
    plot_complexity("Root Mean Squared Error (Horizons Average)", complexity_data, "rmse")
    plot_complexity("Average Training Time (secs/epoch)", complexity_data, "time")
    plot_complexity("Training Time per Node (seconds)", complexity_data, "time-per-node")
    plot_complexity("Average GPU % Used", complexity_data, "gpu")
    plot_complexity("Average (Node) Degree", complexity_data, "neighbors")
    plot_complexity("Clustering Coefficient", complexity_data, "neigh_ratio")


def stand_dev(data, experiment, dataset_name, param):
    data2 = data[~((data['Type'] == "large") & (data["Experiment"] == "Experiment 2"))
                 & (data['Type'] != "comparison")].copy()
    filtered = data2[(data2['Dataset'] == dataset_name) & ((data2['Experiment'] == experiment) | (data2['Type'] == "original"))]
    print(dataset_name, "-", experiment, filtered[param].std())


def lineplots(runs):
    datasets = ["METR-LA", "PEMS-BAY"]
    metric = "Test MAE (AVG)"
    metric_name = "Mean Absolute Error (Horizon's Average)"

    # Call the function
    plot_scalability_lineplots(metric, runs, datasets, "combined", metric_name)
    plot_complexity_lineplots(metric, runs, datasets, "combined", metric_name)

    # plot_performance("Test MAE (AVG)", runs, dataset, "mae", "Test MAE (Horizon's Average)")
    # plot_performance("Test RMSE (AVG)", runs, dataset, "rmse", "Test RMSE (Horizon's Average)")


def large_graphs(data):
    data2 = data[~((data['Type'] == "large") & (data["Experiment"] == "Experiment 2"))
                 & (data['Type'] != "comparison")]# & ~((data["Dataset"] == "PEMS-BAY") & (data["Experiment"] == "Experiment 2"))].copy()
    plot_line_graph(data2, "Mean Absolute Error (Horizons Average)", "mae")
    plot_line_graph(data2, "Root Mean Squared Error (Horizons Average)", "rmse")
    plot_line_graph(data2, "Average Training Time (secs/epoch)", "time")
    plot_line_graph(data2, "Training Time per Node (seconds)", "time-per-node")
    plot_line_graph(data2, "Inference Time per Node (seconds)", "inference-per-node")
    plot_line_graph(data2, "Average GPU % Used", "gpu")
    plot_line_graph(data2, "Average (Node) Degree", "neighbors")
    plot_line_graph(data2, "Clustering Coefficient", "neigh_ratio")
    plot_line_graph(data2, "Edges", "edges")
    plot_line_graph(data2, "Missing values %", "missing-values")



if __name__ == '__main__':
    wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    # update_wandb(runs)
    # analyze_adj_mx(runs)
    data = get_wandb_df(runs)
    data = data[data['Group'] == "1"]


    complexity(data)
    scalability(data)
    # large_graphs(data)
    # lineplots(runs)

    # stand_dev(data, "Experiment 1", "METR-LA", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data, "Experiment 1", "PEMS-BAY", "Mean Absolute Error (Horizons Average)")
    #
    # stand_dev(data, "Experiment 2", "METR-LA", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data, "Experiment 2", "PEMS-BAY", "Mean Absolute Error (Horizons Average)")

