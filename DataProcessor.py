import itertools
from tokenize import String

import folium
import numpy as np
import pandas as pd
from haversine import haversine, Unit, haversine_vector
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist


def read_ids_to_array(file_path):
    with open(file_path, 'r') as file:
        id_list = file.read().strip().split(',')
    return id_list


def get_in_box(sensor_locs, coordinates_of_box):
    return sensor_locs[(sensor_locs.latitude >= coordinates_of_box[3])
                       & (sensor_locs.latitude <= coordinates_of_box[0])
                       & (sensor_locs.longitude >= coordinates_of_box[1])
                       & (sensor_locs.longitude <= coordinates_of_box[2])]


def box_density(points):
    return len(points) / get_area(points)  # in km^2


def get_area(points):
    min_lat = points['latitude'].min()
    max_lat = points['latitude'].max()
    min_lon = points['longitude'].min()
    max_lon = points['longitude'].max()

    width = haversine((min_lat, min_lon), (min_lat, max_lon), unit=Unit.KILOMETERS)
    height = haversine((min_lat, min_lon), (max_lat, min_lon), unit=Unit.KILOMETERS)
    return width * height


def sample_random_points_with_density(points, desired_density):
    num_points = int(round(desired_density * get_area(points)))
    sampled_points = points.sample(n=num_points)
    return sampled_points


def pairwise_distances(points_df, indices):
    distances = {}
    for i, j in itertools.combinations(indices, 2):
        distance = haversine((points_df.at[i, 'latitude'], points_df.at[i, 'longitude']),
                             (points_df.at[j, 'latitude'], points_df.at[j, 'longitude']),
                             unit=Unit.KILOMETERS)
        distances[(i, j)] = distance
    return distances


def select_scattered_points2(points, desired_density):
    num_points = int(round(desired_density * get_area(points)))
    combinations = itertools.combinations(points.index, num_points)
    max_distance = 0
    selected_indices = None

    # Pre-calculate the distance matrix
    matrix = pairwise_distances(points, points.index)
    print("done with matrix")

    for indices in combinations:
        distances = [matrix[(i, j)] for i, j in itertools.combinations(indices, 2)]
        non_zero_distances = [dist for dist in distances if dist != 0]
        min_distance = min(non_zero_distances)
        if min_distance > max_distance:
            max_distance = min_distance
            selected_indices = indices
    print(selected_indices)
    return points.loc[selected_indices, :]


class DataProcessor:
    def __init__(self, data_option, new_h5_filename, new_W_filename, sensor_locations_file):
        self.sensor_locations_file = sensor_locations_file
        self.dataset_name = data_option[0]
        self.sensor_ids_file = data_option[1]
        self.dataset_file = data_option[2]
        self.coordinates = data_option[3]
        self.W_file = data_option[4]
        self.coordinates_bigger = data_option[5]
        self.new_h5_filename = new_h5_filename
        self.new_W_filename = new_W_filename
        self.this_map = folium.Map(prefer_canvas=True)

    def process_data(self):
        sensor_locations = pd.read_csv("../Datasets/" + self.dataset_name + "/" + self.sensor_locations_file, index_col=0)
        w_matrix = pd.read_csv("../Datasets/" + self.dataset_name + "/" + self.W_file)
        data = pd.read_hdf("../Datasets/" + self.dataset_name + "/" + self.dataset_file)

        in_box = get_in_box(sensor_locations, self.coordinates)
        in_comp_box = get_in_box(sensor_locations, self.coordinates_bigger)

        out_of_box = sensor_locations[(sensor_locations.latitude < self.coordinates[3])
                                      | (sensor_locations.latitude > self.coordinates[0])
                                      | (sensor_locations.longitude < self.coordinates[1])
                                      | (sensor_locations.longitude > self.coordinates[2])]

        # ids = in_box.sensor_id.tolist()  # the sensors we train and test with
        indices = in_box.index.tolist()  # the indices of the sensors we train and test with
        # save new subset of data
        filtered_dataset = data.iloc[:, indices]
        filtered_dataset.to_hdf("../Datasets/" + self.dataset_name + "/" + self.new_h5_filename, key='subregion_test', mode='w')
        filtered_w_matrix = w_matrix.iloc[indices, indices]
        filtered_w_matrix.to_csv("../Datasets/" + self.dataset_name + "/" + self.new_W_filename, index=False)

        return in_box, in_comp_box, out_of_box

    def plotDot(self, point, color):
        folium.CircleMarker(location=[point.latitude, point.longitude], radius=8, color=color, stroke=False, fill=True,
                            fill_opacity=0.8, opacity=1, popup=point.sensor_id, fill_color=color).add_to(self.this_map)

    def plot_data(self, name: String, in_box, in_comp_box, out_of_box):
        out_of_box.apply(self.plotDot, axis=1, args=("#000000",))
        in_comp_box.apply(self.plotDot, axis=1, args=("#0000FF",))
        in_box.apply(self.plotDot, axis=1, args=("#FF0000",))
        self.this_map.fit_bounds(self.this_map.get_bounds())
        self.this_map.save(name)
