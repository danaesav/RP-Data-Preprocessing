import string

import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

scalability_ranking = ["(4) Comparison", "(1) Small", "(2) Large", "(3) Original"]
scenario_ranking = ["Comparison", "Small", "Large", "Original"]
complexity_ranking = ["25", "50", "75", "100"]
suffixes = ["100", "075", "050", "025"]


def plot_scalability_lineplots(y, runs, datasets, t, metric_name):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 4))  # 1 row, num_datasets columns

    if num_datasets == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        hist_list = {}
        for run in runs:
            if run.config["Size"] == "100" and run.config["Type"] in ["comparison", "small", "large", "original"] and \
                    run.config["Experiment"] != "Experiment 2" and run.config["Dataset"] == dataset:
                hist_list[run.config["Scenario"]] = run.history(
                    keys=["Test MAE (AVG)", "Test RMSE (AVG)", "Test MAPE (AVG)", "_runtime"])

        plot_data = []
        for run_name, history in hist_list.items():
            df = pd.DataFrame(history)
            df['Run'] = run_name
            df['_runtime'] = df['_runtime'] / 60
            plot_data.append(df)

        plot_data = pd.concat(plot_data)

        sns.lineplot(data=plot_data, x='_runtime', y=y, hue='Run', hue_order=scalability_ranking, ax=ax)
        ax.set_xlabel('Runtime in minutes', fontsize=20)
        ax.set_ylabel('')
        ax.set_title(dataset, fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.get_legend().remove()
        ax.grid(True, which="both", ls="--")
        if i == 0:
            ax.set_ylabel(metric_name, fontsize=16)

    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    for line in handles:
        line.set_linewidth(5)
    legend_ax.legend(handles, scalability_ranking, loc='upper center',
                     bbox_to_anchor=(0.5, -0.2), ncol=4, title="Scenario", title_fontsize=20, fontsize=15)

    plt.tight_layout(pad=2.5)  # Adjust the layout to reserve space for the legend
    plt.subplots_adjust(bottom=0.2, top=1.2)
    plt.savefig("figures/lineplot-scalability-" + t + ".png", bbox_inches='tight')
    plt.show()


def plot_scalability_lineplot(y, runs, dataset, t, metric_name):
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

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='_runtime', y=y, hue='Run', hue_order=scalability_ranking)
    plt.xlabel('Runtime in minutes')
    plt.ylabel(metric_name)
    plt.title(dataset + ': ' + metric_name + ' vs Runtime')
    plt.legend(title='Scenario')
    plt.savefig("figures/" + dataset + "/lineplot-scalability-" + t + "-" + dataset.lower() + ".png",
                bbox_inches='tight')
    plt.show()


def plot_scalability(y, data, t):
    filtered_data = data[
        (data['Size'] == "100") & (data['Type'].isin(["small", "large", "original", "comparison"]))].copy()
    filtered_data['Type'] = filtered_data['Type'].map(lambda x: x.capitalize())
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Type", y=y, order=scenario_ranking, hue="Dataset", errorbar=lambda x: (x.min(), x.max()),
        hue_order=["PEMS-BAY", "METR-LA"], palette="dark",
        height=5.5,
        legend_out=False
    )
    g.set_axis_labels("Scenario", y, fontsize=16)
    g.fig.suptitle(y, y=1.05, fontsize=18)
    ax = g.facet_axis(0, 0)
    ax.set_xlabel("Scenario", fontsize=18)
    ax.set_ylabel(y, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=17)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.1f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge', fontsize=13, rotation=30)
    plt.savefig("figures/scalability/scalability-" + t + ".pdf", bbox_inches='tight')
    plt.show()


def plot_complexity(y, data, t):
    data2 = data[data['Type'] == "large"].copy()
    data2['Size'] = data2['Size'].map(lambda x: x[1:] if x[0] != '1' else x)
    g = sns.catplot(
        data=data2, kind="bar",
        x="Size", y=y, order=complexity_ranking, legend_out=False, hue="Dataset", hue_order=["PEMS-BAY", "METR-LA"],
        palette="dark", height=5
    )
    g.set_axis_labels("Proportion %", y, fontsize=16)
    g.fig.suptitle(y, y=1.05, fontsize=18)
    plt.ylim(bottom=0, top=1.3 * data2[y].max())

    ax = g.facet_axis(0, 0)
    ax.set_xlabel("Proportion %", fontsize=18)
    ax.set_ylabel(y, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.1f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge', fontsize=13, rotation=30, padding=5)

    plt.savefig("figures/complexity/complexity-" + t + ".pdf", bbox_inches='tight')
    plt.show()


def plot_complexity_lineplots(y, runs, datasets, t, metric_name):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 4))  # 1 row, num_datasets columns

    if num_datasets == 1:
        axes = [axes]
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        hist_list = {}
        for run in runs:
            if (run.config["Type"] == "large" and run.config["Dataset"] == dataset):
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

        sns.lineplot(data=plot_data, x='_runtime', y=y, hue='Run', hue_order=complexity_ranking, ax=ax)
        ax.set_title(dataset + ' (Scenario 2)', fontsize=22)
        ax.set_xlabel('Runtime in minutes', fontsize=20)
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.get_legend().remove()
        ax.grid(True, which="both", ls="--")
        if i == 0:
            ax.set_ylabel(metric_name, fontsize=16)

    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    for line in handles:
        line.set_linewidth(5)
    legend_ax.legend(handles, complexity_ranking, loc='upper center',
                     bbox_to_anchor=(0.5, -0.2), ncol=4, title="Proportion %", title_fontsize=20, fontsize=15)

    plt.tight_layout(pad=2.5)  # Adjust the layout to reserve space for the legend
    plt.subplots_adjust(bottom=0.2, top=1.2)

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
        ('Average (Node) Degree', 'summary', 'Average Node Neighbors'),
        ('Clustering Coefficient', 'summary', 'Average Neighbors Ratio'),
        ('Edges', 'summary', 'Edges'),
        ('Group', 'config', 'Group'),
        ('Nodes', 'config', 'Nodes'),
        ('Type', 'config', 'Type'),
        ('Experiment', 'config', 'Experiment'),
        ('Scenario', 'config', 'Scenario'),
        ('Size', 'config', 'Size'),
        ('Dataset', 'config', 'Dataset'),
        ('Missing values %', 'config', 'Missing values %')
    ]

    data = {
        col_name: [getattr(run, attr)[key] for run in runs]
        for col_name, attr, key in columns
    }

    return pd.DataFrame(data)


def plot_line_graph(data, y, file):
    plt.figure(figsize=(8, 6))

    sns.lineplot(data=data, x='Nodes', y=y, hue="Dataset", hue_order=["PEMS-BAY", "METR-LA"],
                 style_order=["Experiment 1", "Experiment 2"], style="Experiment", markers=['o', 'v'], markersize=14)

    plt.xlabel('Road Network Size (#Nodes)', fontsize=20)
    plt.ylabel(y, fontsize=20)
    plt.title(y, fontsize=24)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(title_fontsize='14', fontsize='14')
    # plt.ylim(0, 100)
    plt.savefig("figures/" + file + "_vs_nodes.pdf", bbox_inches='tight')
    plt.show()


# OTHER PLOTTING FUNCTIONS NOT USED (ONLY TO COMBINE PLOTS)

def concat_images(option):
    pemsbay_images = []
    metrla_images = []
    for suffix in suffixes[int(len(suffixes)):]:
        pemsbay_images.append(cv2.imread(f"images/PEMS-BAY/pems-bay-original-{suffix}.png"))
        metrla_images.append(cv2.imread(f"images/METR-LA/metr-la-original-{suffix}.png"))
    pemsbay_image = cv2.vconcat(pemsbay_images)
    metrla_image = cv2.vconcat(metrla_images)
    cv2.imwrite(f"images/METR-LA/metr-la-subsets-exp-2.png", metrla_image)
    cv2.imwrite(f"images/PEMS-BAY/pems-bay-subsets-exp-2.png", pemsbay_image)
    # cv2.imshow('sea_image.jpg', im_v)
