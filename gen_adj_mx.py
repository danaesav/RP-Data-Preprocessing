from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import scipy.sparse as sp
import wandb


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

    wandb.login(key='c273430a11bf8ecb5b86af0f5a16005fc5f2c094')
    api = wandb.Api()
    runs = api.runs("traffic-forecasting-gnns-rp/D2STGNN-final")
    for run in runs:
        # name = run.config['Dataset'].lower() + "-" + run.config['Type'] + "-" + run.config['Size']
        run.summary["Average Node Neighbors"] = node_neighbors[run.name]
        run.summary["Average Neighbors Ratio"] = percentage_neighbors[run.name]
        run.update()
