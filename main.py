from DataProcessor import DataProcessor, box_density, \
    select_scattered_points2

metrla_box_coordinates = [34.174317, -118.409044, -118.345701, 34.131097]
metrla_box_coordinates_bigger = [34.18227, -118.511454, -118.345701, 34.131097]
pemsbay_box_coordinates = [37.378342, -121.932272, -121.894512, 37.350266]
pemsbay_box_coordinates_bigger = [37.393346, -121.952686, -121.873484, 37.329885]
metrla = ["METRLA", "metr_ids.txt", "metr-la.h5", metrla_box_coordinates_bigger, "W_metrla.csv", metrla_box_coordinates_bigger]
pemsbay = ["PEMSBAY", "pemsbay_ids.txt", "pems-bay.h5", pemsbay_box_coordinates_bigger, "W_pemsbay.csv",
           pemsbay_box_coordinates_bigger]
sensor_locations_file = "graph_sensor_locations.csv"

data_option = metrla
new_h5_filename = "new_h5.h5"
new_W_filename = "new_W.h5"

if __name__ == '__main__':
    processor = DataProcessor(data_option, new_h5_filename, new_W_filename, sensor_locations_file)
    within_box, in_comparison_box, outside_of_box = processor.process_data()
    p = select_scattered_points2(within_box, box_density(within_box)/4)
    processor.plot_data("map.html", p, within_box, outside_of_box)
