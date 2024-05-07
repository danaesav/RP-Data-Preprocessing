import io
import itertools
from tokenize import String

import folium
import pandas as pd
from PIL import Image
from haversine import haversine, Unit
from scipy.spatial import ConvexHull


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
    def __init__(self, data_option, sensor_locations_file):
        self.sensor_locations_file = sensor_locations_file
        self.dataset_name = data_option[0]
        self.sensor_ids_file = data_option[1]
        self.dataset_file = data_option[2]
        self.coordinates = data_option[3]
        self.dataset_file_name = data_option[4]
        self.coordinates_bigger = data_option[5]
        self.this_map = folium.Map(prefer_canvas=True, zoom_start=50)

    def process_data(self):
        sensor_locations = pd.read_csv("../Datasets/" + self.dataset_name + "/" + self.sensor_locations_file,
                                       index_col=0)

        in_box = get_in_box(sensor_locations, self.coordinates)
        in_comp_box = get_in_box(sensor_locations, self.coordinates_bigger)

        out_of_box = sensor_locations[(sensor_locations.latitude < self.coordinates[3])
                                      | (sensor_locations.latitude > self.coordinates[0])
                                      | (sensor_locations.longitude < self.coordinates[1])
                                      | (sensor_locations.longitude > self.coordinates[2])]

        return in_box, in_comp_box, out_of_box

    def save_filtered_data(self, in_box, num_points, filename):
        data = pd.read_hdf("../Datasets/" + self.dataset_name + "/" + self.dataset_file + ".h5")

        # ids = in_box.sensor_id.tolist()  # the sensors we train and test with
        # indices = in_box.index.tolist()  # the indices of the sensors we train and test with

        scattered_points = select_scattered_points(in_box, num_points)
        indices = scattered_points.index.tolist()


        # save new subset of data
        filtered_dataset = data.iloc[:, indices]
        # filtered_dataset.to_hdf("../Datasets/" + self.dataset_name + "/" + filename+".h5", key='subregion_test', mode='w')
        # filtered_dataset.to_hdf("../D2STGNN-github/datasets/raw_data/" + self.dataset_file_name + "/" + filename + ".h5", key='subregion_test',
        #                         mode='w')

        # with open("indices/"+filename + ".txt", 'w') as file:
        #     file.write(f"{len(indices)}\n")
        #     file.write(','.join(map(str, indices)))

        return scattered_points

    def reset_map(self):
        self.this_map = folium.Map(prefer_canvas=True, zoom_start=50)

    def plotDot(self, point, color):
        folium.CircleMarker(location=[point.latitude, point.longitude], radius=8, color=color, stroke=False, fill=True,
                            fill_opacity=0.8, opacity=1, popup=point.sensor_id, fill_color=color).add_to(self.this_map)

    def plot_data(self, name: String, in_box, in_comp_box, out_of_box):
        out_of_box.apply(self.plotDot, axis=1, args=("#000000",))
        in_comp_box.apply(self.plotDot, axis=1, args=("#0000FF",))
        in_box.apply(self.plotDot, axis=1, args=("#FF0000",))
        self.this_map.fit_bounds(self.this_map.get_bounds())

        # TO SAVE
        # self.this_map.save("html/" + name + ".html")
        # img_data = self.this_map._to_png(5)
        # img = Image.open(io.BytesIO(img_data))
        # img.save("images/" + name + ".png")
        # self.reset_map()
