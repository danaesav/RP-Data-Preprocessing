import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ttest_rel, ttest_ind, f_oneway
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from DataProcessor import DataProcessor, select_scattered_points
from gen_adj_mx import analyze_adj_mx, update_wandb
from generator import generate_h5_files, generate_adj_mxs, generate_plots, save_adj_mx
from plotting import plot_scalability, plot_complexity, get_wandb_df, plot_performance, plot_scalability2, \
    plot_performance2, plot_performance_complexity

metrla_box_coordinates = [34.188469, -118.509482, -118.439572, 34.132489]
metrla_box_coordinates_bigger = [34.18227, -118.511454, -118.345701, 34.131097]
metrla_box_coordinates_even_bigger = [34.18227, -118.511454, -118.211559, 34.131097]

metrla_box_coordinates_comparison = [34.097491, -118.265575, -118.207500, 34.048523]

pemsbay_box_coordinates = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_bigger = [37.393346, -121.952686, -121.873484, 37.329885]
pemsbay_box_coordinates_even_bigger = [37.414975, -121.987476, -121.880046, 37.313831]
pemsbay_box_coordinates_gigantic = [37.429789, -122.020961, -121.830322, 37.290955]
pemsbay_box_coordinates_comparison = [37.421656, -122.085214, -122.038178, 37.378017]

# [road_distance_small, sensor_ids_file, dataset_file, coordinates, dataset_name, coordinates_bigger, distances_filename, road_distance_large]
metrla = ["METRLA", "metr_ids.txt", "metr-la", metrla_box_coordinates, "METR-LA", metrla_box_coordinates_even_bigger,
          "distances_la_2012.csv"]
metrla2 = ["METRLA", "metr_ids.txt", "metr-la", metrla_box_coordinates_comparison, "METR-LA", metrla_box_coordinates_bigger,
           "distances_la_2012.csv"]
pemsbay = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay", pemsbay_box_coordinates_even_bigger, "PEMS-BAY",
           pemsbay_box_coordinates_gigantic, "distances_bay_2017.csv"]
pemsbay2 = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay", pemsbay_box_coordinates_comparison, "PEMS-BAY",
            pemsbay_box_coordinates_bigger, "distances_bay_2017.csv"]

sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 0.25]
suffixes = ["100", "075", "050", "025"]

data_option = metrla
h5_filename = data_option[2]
distances_filename = data_option[6]




def scalability(scalability_data):
    # plot_scalability("Missing values %", scalability_data, "missing_values")
    plot_scalability("Mean Absolute Error (Horizons Average)", scalability_data, "mae")
    # plot_scalability("Root Mean Squared Error (Horizons Average)", scalability_data, "rmse")
    # plot_scalability("Average Training Time (secs/epoch)", scalability_data, "time")
    # plot_scalability("Training Time per Node (seconds)", scalability_data, "time-per-node")
    # plot_scalability("Average GPU % Used", scalability_data, "gpu")
    # plot_scalability("Average Number of Node Neighbors", scalability_data, "neighbors")
    # plot_scalability("Average Neighbors Ratio", scalability_data, "neigh_ratio")
    # plot_scalability("Edges", scalability_data, "edges")

    # y_values = ["Average Training Time per Node (seconds)", "Average Number of Node Neighbors", "Average GPU % Used"]
    # plot_scalability2(y_values, scalability_data, "combined")


def complexity(complexity_data):
    # plot_complexity("Mean Absolute Error (Horizons Average)", complexity_data, "mae")
    # plot_complexity("Root Mean Squared Error (Horizons Average)", complexity_data, "rmse")
    # plot_complexity("Average Training Time (secs/epoch)", complexity_data, "time")
    plot_complexity("Training Time per Node (seconds)", complexity_data, "time-per-node")
    plot_complexity("Average GPU % Used", complexity_data, "gpu")
    # plot_complexity("Average Number of Node Neighbors", complexity_data, "neighbors")
    # plot_complexity("Average Neighbors Ratio", complexity_data, "neigh_ratio")
    # plot_complexity("Edges", complexity_data, dataset, "edges")
    # plot_complexity("Missing values %", complexity_data, "missing_values")


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


def linear_regression_stats(x, y):
    X = sm.add_constant(x)  # Add a constant term for the intercept
    model = sm.OLS(y, X).fit()
    return model.summary()


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
    # update_wandb()
    # analyze_adj_mx()
    # wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    # api = wandb.Api()
    # runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    # data = get_wandb_df(runs)
    # data2 = data[data['Size'] != "030"].copy()
    # data2['Size'] = data2['Size'].map(lambda x: float(x[1:]) if x[0] != '1' else float(x))

    processor = DataProcessor(metrla, sensor_locations_file)
    in_box, in_comp_box, out_of_box = processor.process_data()
    # print(len(in_comp_box)) # 102 sensors
    processor.save_filtered_data(in_comp_box, len(in_comp_box), metrla[2] + "-huge-100")
    save_adj_mx(metrla[4] + "/" + f"{metrla[2]}-huge-100", metrla)
    # processor.plot_data("metr-la-huge-100", in_comp_box, in_comp_box, out_of_box)

    processor2 = DataProcessor(pemsbay, sensor_locations_file)
    in_box2, in_comp_box2, out_of_box2 = processor2.process_data()
    processor2.save_filtered_data(in_box2, len(in_box2), pemsbay[2] + "-huge-100")
    processor2.save_filtered_data(in_comp_box2, len(in_comp_box2), pemsbay[2] + "-gigantic-100")
    # print(len(in_box2)) # 101 sensors
    # print(len(in_comp_box2)) #207 sensors
    save_adj_mx(pemsbay[4] + "/" + f"{pemsbay[2]}-huge-100", pemsbay)
    save_adj_mx(pemsbay[4] + "/" + f"{pemsbay[2]}-gigantic-100", pemsbay)
    # processor2.plot_data("pems-bay-gigantic-100", in_box2, in_comp_box2, out_of_box2)



    # scattered_points = select_scattered_points(in_box, len(in_box))
    # processor.plot_data("test", in_box, in_comp_box, out_of_box)


    # stand_dev(data2, "large", "METR-LA", "Mean Absolute Error (Horizons Average)")
    # stand_dev(data2, "large", "PEMS-BAY", "Mean Absolute Error (Horizons Average)")

    # some_plot()

    # performance(runs)
    # scalability(data)
    # complexity(data)

    # data2 = data2[data2['Dataset'] == "PEMS-BAY"]
    # data2 = data2[data2['Dataset'] == "METR-LA"]

    # statistics(data2)
