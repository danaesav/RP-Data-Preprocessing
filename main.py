import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ttest_ind

from DataProcessor import DataProcessor

from plotting import plot_scalability, plot_complexity, plot_performance2

metrla_box_coordinates_level_1_25 = [34.188469, -118.509482, -118.439572, 34.132489]
metrla_box_coordinates_level_2_50 = [34.18227, -118.511454, -118.345701, 34.131097]
metrla_box_coordinates_level_3_100 = [34.18227, -118.511454, -118.211559, 34.131097]
metrla_box_coordinates_comparison = [34.097491, -118.265575, -118.207500, 34.048523]
metrla_coordinates = [metrla_box_coordinates_level_1_25, metrla_box_coordinates_level_2_50,
                      metrla_box_coordinates_level_3_100, metrla_box_coordinates_comparison]

pemsbay_box_coordinates_level_1_25 = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_level_2_50 = [37.393346, -121.952686, -121.873484, 37.329885]
pemsbay_box_coordinates_level_3_100 = [37.414975, -121.987476, -121.880046, 37.313831]
pemsbay_box_coordinates_level_4_200 = [37.429789, -122.020961, -121.830322, 37.290955]
pemsbay_box_coordinates_comparison = [37.421656, -122.085214, -122.038178, 37.378017]
pemsbay_coordinates = [pemsbay_box_coordinates_level_1_25, pemsbay_box_coordinates_level_2_50,
                       pemsbay_box_coordinates_level_3_100, pemsbay_box_coordinates_level_4_200,
                       pemsbay_box_coordinates_comparison]

# [dataset_name, sensor_ids_file, distances_filename, coordinates]
metrla = ["METR-LA", "metr_ids.txt", "distances_la_2012.csv", metrla_coordinates]
pemsbay = ["PEMS-BAY", "pemsbay_ids.txt", "distances_bay_2017.csv", pemsbay_coordinates]
proportions = [1, 0.75, 0.5, 0.25]
suffixes = ["100", "075", "050", "025"]

data_option = metrla
filename_start = data_option[0].lower()



def scalability(scalability_data):
    plot_scalability("Mean Absolute Error (Horizons Average)", scalability_data, "mae")
    # plot_scalability("Root Mean Squared Error (Horizons Average)", scalability_data, "rmse")
    # plot_scalability("Average Training Time (secs/epoch)", scalability_data, "time")
    # plot_scalability("Training Time per Node (seconds)", scalability_data, "time-per-node")
    # plot_scalability("Average GPU % Used", scalability_data, "gpu")
    # plot_scalability("Average Number of Node Neighbors", scalability_data, "neighbors")
    # plot_scalability("Average Neighbors Ratio", scalability_data, "neigh_ratio")

    # y_values = ["Average Training Time per Node (seconds)", "Average Number of Node Neighbors", "Average GPU % Used"]
    # plot_scalability2(y_values, scalability_data, "combined")


def complexity(complexity_data):
    plot_complexity("Mean Absolute Error (Horizons Average)", complexity_data, "mae")
    # plot_complexity("Root Mean Squared Error (Horizons Average)", complexity_data, "rmse")
    # plot_complexity("Average Training Time (secs/epoch)", complexity_data, "time")
    # plot_complexity("Training Time per Node (seconds)", complexity_data, "time-per-node")
    # plot_complexity("Average GPU % Used", complexity_data, "gpu")
    # plot_complexity("Average Number of Node Neighbors", complexity_data, "neighbors")
    # plot_complexity("Average Neighbors Ratio", complexity_data, "neigh_ratio")


def stand_dev(dataset, typ, dataset_name, param):
    filter_type = dataset[dataset['Scenario'] == typ]
    filtered = filter_type[filter_type['Dataset'] == dataset_name]
    print(dataset_name,"-", typ, filtered[param].std())



def performance(runs):
    # metrics = [("Test MAE (AVG)", "Test MAE (Horizon's Average)"), ("Test RMSE (AVG)", "Test RMSE (Horizon's Average)")]

    datasets = ["METR-LA", "PEMS-BAY"]
    metric = "Test MAE (AVG)"
    metric_name = "Test Mean Absolute Error (Horizon's Average)"

    # Call the function
    plot_performance2(metric, runs, datasets, "combined", metric_name)
    # plot_performance_complexity(metric, runs, datasets, "combined", metric_name)

    # plot_performance("Test MAE (AVG)", runs, dataset, "mae", "Test MAE (Horizon's Average)")
    # plot_performance("Test RMSE (AVG)", runs, dataset, "rmse", "Test RMSE (Horizon's Average)")

def statistics(data2):
    mapping = {"small": 1, "large": 2, "original": 3}
    data2['Scenario_encoded'] = data2['Scenario'].map(mapping)

    print(len(data2['Missing values %']))
    correlation_coefficient_missing, p_value_mae_missing = pearsonr(data2['Mean Absolute Error (Horizons Average)'],
                                                                    data2['Missing values %'])
    print(len(data2['Nodes']))
    correlation_coefficient_gpu, p_value_gpu = pearsonr(data2['Nodes'], data2['Average GPU % Used'])
    # correlation_coefficient_training, p_value_training = pearsonr(data2['Average Training Time per Node (seconds)'],
    #                                                               data2['Nodes'])
    data_metr = data2[(data2['Dataset'] == "METR-LA") & (data2['Scenario'] != 'comparison') & (data2['Size'] == 100)]
    data_bay = data2[(data2['Dataset'] == "PEMS-BAY") & (data2['Scenario'] != 'comparison') & (data2['Size'] == 100)]

    correlation_coefficient_nodes_bay, p_value_mae_nodes_bay = pearsonr(
        data_bay['Mean Absolute Error (Horizons Average)'], data_bay['Scenario_encoded'])
    correlation_coefficient_nodes_metr, p_value_mae_nodes_metr = pearsonr(
        data_metr['Mean Absolute Error (Horizons Average)'], data_metr['Scenario_encoded'])

    data_metr = data2[(data2['Dataset'] == "METR-LA") & (data2['Scenario'] == 'large')]
    data_bay = data2[(data2['Dataset'] == "PEMS-BAY") & (data2['Scenario'] == 'large')]

    correlation_coefficient_complexity_bay, p_value_mae_complexit_bay = pearsonr(
        data_bay['Mean Absolute Error (Horizons Average)'], data_bay['Size'])
    correlation_coefficient_complexit_metr, p_value_mae_complexit_metr = pearsonr(
        data_metr['Mean Absolute Error (Horizons Average)'], data_metr['Size'])

    print(
        f"Correlation coefficient between MAE and missing values: {correlation_coefficient_missing}, P-value: {p_value_mae_missing}")
    print(f"Correlation coefficient between nodes and GPU usage: {correlation_coefficient_gpu}, P-value: {p_value_gpu}")
    # print(
    #     f"Correlation coefficient between training time and nodes: {correlation_coefficient_training}, P-value: {p_value_training}")
    print(
        f"Correlation coefficient between MAE and scenarios (METR-LA): {correlation_coefficient_nodes_metr}, P-value: {p_value_mae_nodes_metr}")
    print(
        f"Correlation coefficient between MAE and scenarios (PEMS-BAY): {correlation_coefficient_nodes_bay}, P-value: {p_value_mae_nodes_bay}")

    print(
        f"Correlation coefficient between MAE and complexity (METR-LA): {correlation_coefficient_complexit_metr}, P-value: {p_value_mae_complexit_metr}")
    print(
        f"Correlation coefficient between MAE and complexity (PEMS-BAY): {correlation_coefficient_complexity_bay}, P-value: {p_value_mae_complexit_bay}")

    # We have found a significant positive correlation between the number of nodes and the average GPU usage.
    # We have also found a significant negative correlation between the number of nodes and the average training time per node.
    # We have found a significant positive correlation between the mean absolute error and the missing values percentage.

    # t_statistic, p_value = ttest_rel(data2['Nodes'], data2['Average GPU % Used'])
    data_metr = data2[(data2['Dataset'] == "METR-LA") & (data2['Scenario'] == 'large')]
    data_bay = data2[(data2['Dataset'] == "PEMS-BAY") & (data2['Scenario'] == 'large')]
    t_statistic_2, p_value_2 = ttest_ind(data_metr['Average GPU % Used'], data_bay['Average GPU % Used'])

    sensor_1 = [3.51, 3.33]
    sensor_2 = [4.72, 4.64]

    t_statistic_sensors, p_value_sensors = ttest_ind(sensor_1, sensor_2)
    cor, p = pearsonr(sensor_1, sensor_2)
    print(f"Correlation coefficient between two sensors: {cor}, P-value: {p}")

    print(f"Independent T-test: T-statistic: {t_statistic_2}, P-value: {p_value_2}")
    print(f"Independent sensors T-test: T-statistic: {t_statistic_sensors}, P-value: {p_value_sensors}")


if __name__ == '__main__':
    # wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    # api = wandb.Api()
    # runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    # update_wandb(runs)
    # analyze_adj_mx(runs)
    # data = get_wandb_df(runs)
    # data2['Size'] = data2['Size'].map(lambda x: float(x[1:]) if x[0] != '1' else float(x))

    processor = DataProcessor(metrla)
    points, out_of_box = processor.get_subsets()
    # print(len(in_comp_box)) # 102 sensors
    processor.save_data(points[0], len(points[0]), filename_start + "-huge-100")
    # processor.plot_data("metr-la-all-100", points, out_of_box)

    processor2 = DataProcessor(pemsbay)
    points2, out_of_box2 = processor2.get_subsets()
    processor2.save_data(points[2], len(points[2]), filename_start + "-huge-100")
    processor2.save_data(points[3], len(points[3]), filename_start + "-gigantic-100")
    # print(len(in_box2)) # 101 sensors
    # print(len(in_comp_box2)) #207 sensors
    # processor2.plot_data("pems-bay-all-100", points2, out_of_box2)



    # scattered_points = select_scattered_points(in_box, len(in_box))
    # processor.plot_data("test", in_box, in_comp_box, out_of_box)


    # stand_dev(data2, "large", "METR-LA", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data2, "large", "PEMS-BAY", "Mean Absolute Error (Horizons Average)")


    # performance(runs)
    # scalability(data)
    # complexity(data)

    # data2 = data2[data2['Dataset'] == "PEMS-BAY"]
    # data2 = data2[data2['Dataset'] == "METR-LA"]

    # statistics(data2)
