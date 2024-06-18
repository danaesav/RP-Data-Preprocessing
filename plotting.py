import string

import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

scalability_ranking = ["comparison", "small", "large", "original"]
complexity_ranking = ["25", "50", "75", "100"]
sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 1 / 3, 0.25]
suffixes = ["100", "075", "050", "030", "025"]

def plot_performance2(y, runs, datasets, t, metric_name):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(8 * num_datasets, 5))  # 1 row, num_datasets columns

    if num_datasets == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        hist_list = {}
        for run in runs:
            if run.config["Size"] == "100" and run.config["Dataset"] == dataset:
                hist_list[run.name] = run.history(
                    keys=["Test MAE (AVG)", "Test RMSE (AVG)", "Test MAPE (AVG)", "_runtime"])

        plot_data = []
        for run_name, history in hist_list.items():
            df = pd.DataFrame(history)
            df['Run'] = run_name.split('-')[2].capitalize()
            df['_runtime'] = df['_runtime'] / 60
            plot_data.append(df)

        plot_data = pd.concat(plot_data)

        # Create line plot
        hue_order = ["Comparison", "Small", "Large", "Original"]
        sns.lineplot(data=plot_data, x='_runtime', y=y, hue='Run', hue_order=hue_order, ax=ax)
        ax.set_xlabel('Runtime in minutes')
        ax.set_ylabel(metric_name)
        ax.set_title(dataset + ': ' + metric_name + ' vs Runtime')
        ax.get_legend().remove()

    # Create a separate subplot for the legend below the subplots
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')  # Hide the axes of the legend subplot
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, ["Comparison", "Small", "Large", "Original"], loc='upper center',
                     bbox_to_anchor=(0.5, -0.1), ncol=4, title="Scenario", title_fontsize='large')

    plt.tight_layout(pad=2.0)  # Adjust the layout to reserve space for the legend
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to make space for the legend
    plt.savefig("figures/lineplot-scalability-" + t + ".png", bbox_inches='tight')
    plt.show()


def plot_performance(y, runs, dataset, t, metric_name):
    hist_list = {}
    for run in runs:
        if run.config["Size"] == "100" and run.config["Dataset"] == dataset:
            hist_list[run.name] = run.history(keys=["Test MAE (AVG)", "Test RMSE (AVG)", "Test MAPE (AVG)", "_runtime"])

    plot_data = []
    for run_name, history in hist_list.items():
        df = pd.DataFrame(history)
        df['Run'] = run_name.split('-')[2].capitalize()
        df['_runtime'] = df['_runtime'] / 60
        plot_data.append(df)

    plot_data = pd.concat(plot_data)

    # Create line plot
    hue_order = ["Comparison", "Small", "Large", "Original"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='_runtime', y=y, hue='Run', hue_order=hue_order)
    plt.xlabel('Runtime in minutes')
    plt.ylabel(metric_name)
    plt.title(dataset + ': ' + metric_name + ' vs Runtime')
    plt.legend(title='Scenario')
    plt.savefig("figures/" + dataset + "/lineplot-scalability-" + t + "-" + dataset.lower() + ".png", bbox_inches='tight')
    plt.show()



def plot_scalability(y, data, t):
    filtered_data = data[data['Size'] == "100"]
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Scenario", y=y, order=scalability_ranking, hue="Dataset", hue_order=["PEMS-BAY", "METR-LA"], palette="dark",
        height=5.5,
        legend_out=False  # Ensure the legend is not outside the plot area
    )
    g.set_axis_labels("Scenario", y)
    g.fig.suptitle(y, y=1.05)  # Adjust title position
    ax = g.facet_axis(0, 0)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.2f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge')
    plt.savefig("figures/scalability/" + "scalability-" + t + ".pdf", bbox_inches='tight')
    plt.show()


def plot_complexity(y, data, t):
    data2 = data[data['Size'] != "030"].copy()
    data2['Size'] = data2['Size'].map(lambda x: x[1:] if x[0] != '1' else x)
    filtered_data = data2[data2['Scenario'] == "large"]
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Size", y=y, order=complexity_ranking, legend_out=False, hue="Dataset", hue_order=["PEMS-BAY", "METR-LA"],
        palette="dark", height=5
    )
    g.set_axis_labels("Proportion %", y)
    g.fig.suptitle(y, y=1.05)
    # g.fig.subplots_adjust(top=0.8)  # Adjust the top margin
    plt.ylim(bottom=0, top=1.3 * filtered_data[y].max())

    ax = g.facet_axis(0, 0)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.2f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge')

    # Place the legend inside the plot
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Adjust the bbox_to_anchor as needed

    plt.savefig("figures/complexity/" + "complexity-" + t + ".pdf", bbox_inches='tight')
    plt.show()


def plot_performance_complexity(y, runs, datasets, t, metric_name):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(8 * num_datasets, 5))  # 1 row, num_datasets columns

    if num_datasets == 1:
        axes = [axes]
    hue_order = ["25", "50", "75", "100"]
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        hist_list = {}
        for run in runs:
            if run.config["Size"] != "030" and run.config["Type"] == "large" and run.config["Dataset"] == dataset:
                hist_list[run.name] = run.history(
                    keys=["Test MAE (AVG)", "Test RMSE (AVG)", "Test MAPE (AVG)", "_runtime"])

        plot_data = []
        for run_name, history in hist_list.items():
            df = pd.DataFrame(history)
            size = run_name.split('-')[3]
            df['Run'] = size if size[0] == '1' else size[1:]
            df['_runtime'] = df['_runtime'] / 60
            plot_data.append(df)

        plot_data = pd.concat(plot_data)

        # Create line plot

        sns.lineplot(data=plot_data, x='_runtime', y=y, hue='Run', hue_order=hue_order, ax=ax)
        ax.set_xlabel('Runtime in minutes')
        ax.set_ylabel(metric_name)
        ax.set_title(dataset + ' (Scenario 2): ' + metric_name + ' vs Runtime')
        ax.get_legend().remove()

    # Create a separate subplot for the legend below the subplots
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')  # Hide the axes of the legend subplot
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, hue_order, loc='upper center',
                     bbox_to_anchor=(0.5, -0.1), ncol=4, title="Proportion %", title_fontsize='large')

    plt.tight_layout(pad=2.0)  # Adjust the layout to reserve space for the legend
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to make space for the legend
    plt.savefig("figures/lineplot-complexity-" + t + ".png", bbox_inches='tight')
    plt.show()

def get_wandb_df(runs):
    columns = [
        ('Mean Absolute Error (Horizons Average)', 'summary', 'Test MAE (AVG)'),
        ('Mean Absolute Percentage Error (Horizons Average)', 'summary', 'Test MAPE (AVG)'),
        ('Root Mean Squared Error (Horizons Average)', 'summary', 'Test RMSE (AVG)'),
        ('Average Training Time (secs/epoch)', 'summary', 'AVG Training time secs/epoch'),
        ('Average Inference Time (secs/epoch)', 'summary', 'AVG Inference time secs/epoch'),
        ('Training Time per Node (seconds)', 'summary', 'AVG Training Time/nodes'),
        ('Inference Time per Node (seconds)', 'summary', 'AVG Inference Time/nodes'),
        ('Average GPU % Used', 'summary', 'Average GPU % Usage'),
        ('Average Number of Node Neighbors', 'summary', 'Average Node Neighbors'),
        ('Average Neighbors Ratio', 'summary', 'Average Neighbors Ratio'),
        ('Edges', 'summary', 'Edges'),
        ('Nodes', 'config', 'Nodes'),
        ('Scenario', 'config', 'Type'),
        ('Size', 'config', 'Size'),
        ('Dataset', 'config', 'Dataset'),
        ('Missing values %', 'config', 'Missing values %')
    ]

    data = {
        col_name: [getattr(run, attr)[key] for run in runs]
        for col_name, attr, key in columns
    }

    return pd.DataFrame(data)




# OTHER PLOTTING FUNCTIONS NOT USED (ONLY TO COMBINE PLOTS)

def concat_images(option):
    large_images = []
    small_images = []
    for size, suffix in zip(sizes, suffixes):
        large_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-large-{suffix}.png"))
        small_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-small-{suffix}.png"))
    large_image = cv2.vconcat(large_images)
    small_image = cv2.vconcat(small_images)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-large.png", large_image)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-small.png", small_image)
    # show the output image
    # cv2.imshow('sea_image.jpg', im_v)

def plot_scalability2(y_values, data, t):
    data = data[(data['Size'] == "100")]
    num_plots = len(y_values)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))  # 1 row, num_plots columns

    for i, y in enumerate(y_values):
        ax = axes[i] if num_plots > 1 else axes
        sns.barplot(data=data, x="Scenario", y=y, order=scalability_ranking, hue="Dataset", palette="dark", ax=ax)
        ax.set_xlabel("Scenario")
        ax.set_ylabel(y)
        ax.set_title(y)
        # ax.set_title(f"({chr(97+i)}) {y}")
        ax.text(0.5, -0.1, f"({string.ascii_lowercase[i]})", transform=ax.transAxes, fontsize=12, va='top', ha='right')

        ax.get_legend().remove()
        for i in range(0, len(ax.containers)):
            c = ax.containers[i]
            labels = [f"{val:.2f}" for val in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='edge')

    # Create a separate subplot for the legend below the subplots
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')  # Hide the axes of the legend subplot
    legend_ax.legend(handles=[], labels=[])  # Add an empty legend to reserve space
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, ["PEMS-BAY", "METR-LA"], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                     title="Dataset", title_fontsize='large')  # Add the legend
    plt.setp(legend_ax.get_legend().get_title(), fontsize='large')  # Set legend title font size

    plt.tight_layout()  # Adjust the layout to reserve space for the legend
    plt.subplots_adjust(bottom=0.20)
    plt.savefig("figures/scalability/" + "scalability-" + t + ".png", bbox_inches='tight')
    plt.show()