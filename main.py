from tokenize import String

import pandas as pd
import folium

# Metr-La points
# Top left 37.393346,-121.952686
# Top right 37.393346,-121.873484
# Bottom right 37.329885,-121.873484
# Bottom left 37.329885,-121.952686

# Pems-Bay points
# Top left 37.378342,-121.932272
# Top right 37.378342,-121.894512
# Bottom right 37.350266,-121.894512
# Bottom left 37.350266,-121.932272

# Metr-La bigger box
# Top left 37.393346,-121.952686
metrla_box_coordinates = [34.174317, -118.409044, -118.345701, 34.131097]
metrla_box_coordinates_bigger = [34.18227, -118.511454, -118.345701, 34.131097]
pemsbay_box_coordinates = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_bigger = [37.393346, -121.952686, -121.873484, 37.329885]
metrla = ["METRLA", "metr_ids.txt", "metr-la.h5", metrla_box_coordinates, "W_metrla.csv", metrla_box_coordinates_bigger]
pemsbay = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay.h5", pemsbay_box_coordinates, "W_pemsbay.csv",
           pemsbay_box_coordinates_bigger]
sensor_locations_file = "graph_sensor_locations.csv"

data_option = metrla
new_h5_filename = "new_h5.h5"
new_W_filename = "new_W.h5"

dataset_name = data_option[0]
sensor_ids_file = data_option[1]
dataset_file = data_option[2]
coordinates = data_option[3]
W_file = data_option[4]
coordinates_bigger = data_option[5]
this_map = folium.Map(prefer_canvas=True)


def read_ids_to_array(file_path):
    with open(file_path, 'r') as file:
        # Read the single line containing the IDs, remove any trailing newline characters and split by comma
        id_list = file.read().strip().split(',')
    return id_list


def process_data():
    sensor_locations = pd.read_csv("../Datasets/" + dataset_name + "/" + sensor_locations_file, index_col=0)
    print(sensor_locations)
    w_matrix = pd.read_csv("../Datasets/" + dataset_name + "/" + W_file)
    data = pd.read_hdf("../Datasets/" + dataset_name + "/" + dataset_file)

    in_box = sensor_locations[(sensor_locations.latitude >= coordinates[3])
                              & (sensor_locations.latitude <= coordinates[0])
                              & (sensor_locations.longitude >= coordinates[1])
                              & (sensor_locations.longitude <= coordinates[2])]

    in_comp_box = sensor_locations[(sensor_locations.latitude >= coordinates_bigger[3])
                                   & (sensor_locations.latitude <= coordinates_bigger[0])
                                   & (sensor_locations.longitude >= coordinates_bigger[1])
                                   & (sensor_locations.longitude <= coordinates_bigger[2])]

    out_of_box = sensor_locations[(sensor_locations.latitude < coordinates[3])
                                  | (sensor_locations.latitude > coordinates[0])
                                  | (sensor_locations.longitude < coordinates[1])
                                  | (sensor_locations.longitude > coordinates[2])]

    ids = in_box.sensor_id.tolist()  # the sensors we train and test with
    indices = in_box.index.tolist()  # the indices of the sensors we train and test with

    # save new subset of data
    filtered_dataset = data.iloc[:, indices]
    filtered_dataset.to_hdf("../Datasets/" + dataset_name + "/" + new_h5_filename, key='subregion_test', mode='w')
    filtered_w_matrix = w_matrix.iloc[indices, indices]
    filtered_w_matrix.to_csv("../Datasets/" + dataset_name + "/" + new_W_filename, index=False)

    return in_box, in_comp_box, out_of_box


def plotDot(point, color):
    folium.CircleMarker(location=[point.latitude, point.longitude], radius=8, color=color, stroke=False, fill=True,
                        fill_opacity=0.8, opacity=1, popup=point.sensor_id, fill_color=color).add_to(this_map)


def plot_data(name: String, in_box, in_comp_box, out_of_box):
    out_of_box.apply(plotDot, axis=1, args=("#000000",))
    in_comp_box.apply(plotDot, axis=1, args=("#0000FF",))
    in_box.apply(plotDot, axis=1, args=("#FF0000",))
    this_map.fit_bounds(this_map.get_bounds())
    this_map.save(name)


if __name__ == '__main__':
    within_box, in_comparison_box, outside_of_box = process_data()
    plot_data("map.html", within_box, in_comparison_box, outside_of_box)
