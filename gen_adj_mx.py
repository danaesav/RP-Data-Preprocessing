from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import wandb
def update_wandb():
    wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    for run in runs:
        data1 = None
        if run.config["Type"] == "original":
            data1 = pd.read_hdf("Datasets/" + run.config["Dataset"] + "/" + run.config["Dataset"].lower() + ".h5")
        else:
            data1 = pd.read_hdf("../D2STGNN-github/datasets/raw_data/" + run.config["Dataset"] + "/" + run.name + ".h5")
        num_rows_to_take = int(0.7 * data1.shape[0])
        training_data = data1.iloc[:num_rows_to_take, :]
        zero_count1 = (data1 == 0).sum().sum()
        zero_count_training = (training_data == 0).sum().sum()
        if "small-0" in run.name:
            continue
        print("Percentage of missing values in ", run.name, ": ", zero_count1 / (data1.shape[0] * data1.shape[1]) * 100)
        print("Percentage of missing values in training data: ", zero_count_training / (training_data.shape[0] * training_data.shape[1]) * 100)

        # total_training_time = run.summary['AVG Training time secs/epoch'] * 80
        # total_inference_time = run.summary['AVG Inference time secs/epoch'] * 80
        # run.summary["Total Inference Time"] = total_inference_time
        # run.summary["Total Training Time"] = total_training_time
        # run.summary["Average GPU % Usage"] = np.mean(run.history(stream="events").loc[:, "system.gpu.process.0.gpu"])
        # run.update()
        # run.summary["AVG Training Time/nodes"] = run.summary['Total Training Time'] / run.config['Nodes']
        # run.summary["AVG Inference Time/nodes"] = run.summary['Total Inference Time'] / run.config['Nodes']
        # run.summary["AVG Training Time per Node Per Epoch"] = run.summary['AVG Training time secs/epoch'] / run.config['Nodes']
        # run.config["Missing values %"] = zero_count1 / (data1.shape[0] * data1.shape[1]) * 100
        # run.config["Missing values % in training"] = zero_count1 / (data1.shape[0] * data1.shape[1]) * 100
        # run.update()

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


def transition_matrix(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # P = d_mat.dot(adj)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(file_path):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    return adj, adj_mx


suffixes = ["025", "030", "050", "075", "100"]
types = ["large", "small", "comparison", "original"]
datasets = ["METR-LA", "PEMS-BAY"]


def analyze_adj_mx():
    path = "../D2STGNN-github/datasets/sensor_graph/adj_mxs/"
    node_neighbors = {}
    percentage_neighbors = {}
    edges = {}
    for dataset in datasets:
        dataset_path = path + dataset + "/"
        for t in types:
            for suffix in suffixes:
                name = dataset.lower() + "-" + t + "-" + suffix
                full_path = dataset_path + name + ".pkl"
                if not os.path.isfile(full_path):
                    continue
                adj_mx, adj_ori = load_adj(full_path)
                sigma = np.std(adj_mx)
                threshold_kappa = 0.1
                adj_matrix = np.array(adj_mx)
                edge_weights = np.exp(-(adj_matrix ** 2) / (2 * (sigma ** 2)))
                adj_matrix_thresholded = np.where(adj_matrix <= threshold_kappa, edge_weights, 0)
                for i in range(adj_matrix_thresholded.shape[0]):
                    adj_matrix_thresholded[i, i] = 0
                num_neighbors = np.sum(adj_matrix_thresholded > 0, axis=1)
                node_neighbors[name] = round(np.mean(num_neighbors), 2)
                percentage_neighbors[name] = round((node_neighbors[name] / adj_mx[0].shape[0]) * 100, 2)
                sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(full_path)
                edgess = 0
                for i in range(adj_mx.shape[0]):
                    for j in range(adj_mx.shape[1]):
                        if adj_mx[i, j] != 0:
                            edgess+=1
                edges[name] = edgess


    wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    for run in runs:
        # name = run.config['Dataset'].lower() + "-" + run.config['Type'] + "-" + run.config['Size']
        if run.config['Dataset'] != "PEMS-BAY":
            continue
        run.summary["Average Node Neighbors"] = node_neighbors[run.name]
        run.summary["Average Neighbors Ratio"] = percentage_neighbors[run.name]
        run.summary["Edges"] = edges[run.name]
        print(run.name, edges[run.name])
        run.update()
