import io
import itertools
import os
import pickle

import folium
import pandas as pd
from PIL import Image
from haversine import haversine, Unit
from scipy.spatial import ConvexHull

from gen_adj_mx import get_adjacency_matrix


def save_adj_mx(filename, dataset_name, distances_filename):
    with open("ids/" + dataset_name + "/" + filename + ".txt") as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv("sensor_graph/" + distances_filename, dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
    # Save to pickle file.
    if not os.path.exists("../D2STGNN-github/datasets/sensor_graph/adj_mxs/" + dataset_name):
        os.makedirs("../D2STGNN-github/datasets/sensor_graph/adj_mxs/" + dataset_name)
    with open("../D2STGNN-github/datasets/sensor_graph/adj_mxs/" + dataset_name + "/" + filename + ".pkl", 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)



def read_ids_to_array(file_path):
    with open(file_path, 'r') as file:
        id_list = file.read().strip().split(',')
    return id_list


def get_in_box(sensor_locs, coordinates_of_box):
    return sensor_locs[(sensor_locs.latitude >= coordinates_of_box[3])
                       & (sensor_locs.latitude <= coordinates_of_box[0])
                       & (sensor_locs.longitude >= coordinates_of_box[1])
                       & (sensor_locs.longitude <= coordinates_of_box[2])]


def select_scattered_points(points, num_points):
    # num_points = int(round(desired_density * get_area(points)))
    if num_points >= len(points):
        return points

    # Compute the convex hull of the points
    hull = ConvexHull(points[['latitude', 'longitude']])
    hull_points_indices = hull.vertices

    selected_indices = []
    selected_points = []

    # Select up to num_points from the hull
    for point_index in hull_points_indices:
        if len(selected_points) < num_points:
            selected_indices.append(point_index)
            selected_points.append(points.iloc[point_index])

    # If fewer than num_points are selected from the hull, select additional points from the interior
    while len(selected_points) < num_points:
        # Find the point from the interior that maximizes its distance from the selected points
        max_distance = 0
        max_distance_point_index = None
        for i, point in points.iterrows():
            if i not in selected_indices:
                distances = [haversine((point['latitude'], point['longitude']),
                                       (selected_point['latitude'], selected_point['longitude']),
                                       unit=Unit.KILOMETERS) for selected_point in selected_points]
                min_distance = min(distances)
                if min_distance > max_distance:
                    max_distance = min_distance
                    max_distance_point_index = i

        # Add the selected point to the list
        selected_indices.append(max_distance_point_index)
        selected_points.append(points.loc[max_distance_point_index])

    selected_points_df = pd.DataFrame(selected_points)
    return selected_points_df


class DataProcessor:
    def __init__(self, data_option):
        self.sensor_locations_file = "graph_sensor_locations.csv"
        self.dataset_name = data_option[0]
        self.filename_start = data_option[0].lower()
        self.sensor_ids_file = data_option[1]
        self.distances_filename = data_option[2]
        self.coordinates = data_option[3]
        self.this_map = folium.Map(prefer_canvas=True, zoom_start=50)

    def get_subsets(self):
        sensor_locations = pd.read_csv("Datasets/" + self.dataset_name + "/" + self.sensor_locations_file,
                                       index_col=0)
        points = []
        for level in self.coordinates:
            points_in_level = get_in_box(sensor_locations, level)
            points.append(points_in_level)

        return points, sensor_locations

    def save_data(self, points, num_points, filename):
        data = pd.read_hdf("Datasets/" + self.dataset_name + "/" + self.filename_start + ".h5")

        scattered_points = select_scattered_points(points, num_points)
        ids = [int(x) for x in scattered_points.sensor_id.tolist()]
        indices = scattered_points.index.tolist()

        # save new subset of data
        filtered_dataset = data.iloc[:, indices]
        filtered_dataset.to_hdf("../D2STGNN-github/datasets/raw_data/" + self.dataset_name + "/" + filename + ".h5", key='subregion_test',
                                mode='w')

        with open("ids/" + self.dataset_name + "/" + filename + ".txt", 'w') as file:
            file.write(','.join(map(str, ids)))

        with open("indices/" + self.dataset_name + "/" + filename + ".txt", 'w') as file:
            file.write(f"{len(indices)}\n")
            file.write(','.join(map(str, indices)))

        save_adj_mx(filename, self.dataset_name, self.distances_filename)

        return scattered_points

    def reset_map(self):
        self.this_map = folium.Map(prefer_canvas=True, zoom_start=50)

    def plotDot(self, point, color):
        folium.CircleMarker(location=[point.latitude, point.longitude], radius=8, color=color, stroke=False, fill=True,
                            fill_opacity=0.8, opacity=1, popup=point.sensor_id, fill_color=color).add_to(self.this_map)

    def plot_data(self, name, points, out_of_box):
        # out_of_box.apply(self.plotDot, axis=1, args=("#000000",))
        points.apply(self.plotDot, axis=1, args=("#FF0000",))
        # in_comp_box.apply(self.plotDot, axis=1, args=("#0000FF",))
        # in_bigger.apply(self.plotDot, axis=1, args=("#32cd32",)) #green

        # points[6].apply(self.plotDot, axis=1, args=("#a9a9a9",))


        # red, green, orange, purple, blue (for comparison)
        # colors = ["#FF0000", "#32cd32", "#FFA500", "#800080", "#a9a9a9", "#469990", "#0000FF"]
        #
        # for i in range(1, len(points)):
        #     level = points[len(points) - i - 1] # apply coloring in reverse order
        #     level.apply(self.plotDot, axis=1, args=(colors[len(points) - i - 1],))

        # points[-1].apply(self.plotDot, axis=1, args=("#0000FF",))

        self.this_map.fit_bounds(self.this_map.get_bounds())

        # TO SAVE
        self.this_map.save("html/" + name + ".html")
        img_data = self.this_map._to_png(5)
        img = Image.open(io.BytesIO(img_data))
        img.save("images/" + name + ".png")
        self.reset_map()
