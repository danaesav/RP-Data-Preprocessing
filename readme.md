# Scalability of Graph Neural Networks in Traffic Forecasting

This repository contains the preprocessing code for my bachelor thesis "Scalability of Graph Neural Networks in Traffic Forecasting" at TU Delft, 2024.

## Data
The data used in this project is the METR-LA and PEMS-BAY datasets, which can be downloaded from [this link]().
Only the .h5 files need to be downloaded and should be placed in the `Datasets/{dataset_name}` directory, the rest of the data is included. 

## Sensors
The sensor ids and their indices used in the experiment for each subset can be found in the `ids` and `indixes` directories respectively.
Visualizations of the sensors can be found in the `images` and `html` directories.

## Subsets
The subsets are selected based on box coordinates. All the sensors within that box will be selected. The boxes for the subsets used in the experiment can be found in the `main.py` file. Different sizes (levels) of boxes can be found. Adding a new box is as simple as adding a new entry to the `metrla_coordinates` array.

The `main.py` file contains the code to preprocess the data.
To create the subsets, follow the following code snippet:

```python
processor = DataProcessor(metrla)
subsets, other_points = processor.get_subsets()
processor.save_data(subsets[0], len(subsets[0]), "subset_name")
```

With `subsets[0]` being the subset you want to save based on the order set in the coordinates array.
The `save_data` method will save the sensor ids, indices of the sensors, and the adjacency matrix of the subset. It will also save the subset as a `.h5` file to be used in the D2STGNN model.

To save a proportion of the points for Experiment 2, simply change the `num_points` parameter in the `save_data` method.
E.g. `len(subsets[0]) * 0.5` will save half of the points in the subset.

To plot and save the subsets on an html map and .png file, as shown in the paper, use the following code snippet:

```python
    processor.plot_data("filename", subsets, other_points)
```

## Statistics
To generate the plots of the paper run the `complexity` and `scalability` methods in the `main.py` file. The plots will be saved in the `figures` directory.
Note that the data needs to be in wandb to generate the plots.

