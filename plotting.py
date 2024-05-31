import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

scalability_ranking = ["comparison", "small", "large", "original"]
complexity_ranking = ["025", "030", "050", "075", "100"]


def plot_scalability(y, data, t):
    # data2 = data[data['Type'] != "comparison"]
    filtered_data = data[data['Size'] == "100"]
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Type", y=y, order=scalability_ranking, hue="Dataset", hue_order=["PEMS-BAY", "METR-LA"], palette="dark", height=6
    )
    g.set_axis_labels("Map Area", y)
    g.fig.suptitle(y + " for both datasets")
    g.fig.subplots_adjust(top=0.9)
    ax = g.facet_axis(0, 0)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.2f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge')
    plt.savefig("figures/scalability-" + t + "-" + ".png")
    plt.show()

def plot_complexity(y, data, dataset, t):
    data2 = data[data['Type'].isin(["small", "large"])]
    filtered_data = data2[data2['Dataset'] == dataset]
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Size", y=y, order=complexity_ranking, hue="Type", hue_order=["small", "large"], palette="dark", height=6
    )
    g.set_axis_labels("Map Area", y)
    g.fig.suptitle(y + " for " + dataset)
    g.fig.subplots_adjust(top=0.9)
    ax = g.facet_axis(0, 0)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.2f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge')
    plt.savefig("figures/" + dataset + "/complexity-" + t + "-" + dataset.lower() + ".png")
    plt.show()


def get_wandb_df(runs):
    columns = [
        ('Mean Absolute Error (Horizons Average)', 'summary', 'Test MAE (AVG)'),
        ('Mean Absolute Percentage Error (Horizons Average)', 'summary', 'Test MAPE (AVG)'),
        ('Root Mean Squared Error (Horizons Average)', 'summary', 'Test RMSE (AVG)'),
        ('Average Training Time (secs/epoch)', 'summary', 'AVG Training time secs/epoch'),
        ('Average Inference Time (secs/epoch)', 'summary', 'AVG Inference time secs/epoch'),
        ('Average Training Time/node', 'summary', 'AVG Training Time/nodes'),
        ('Average GPU % Used', 'summary', 'Average GPU % Usage'),
        ('Average Node Neighbors', 'summary', 'Average Node Neighbors'),
        ('Average Neighbors Ratio', 'summary', 'Average Neighbors Ratio'),
        ('Nodes', 'config', 'Nodes'),
        ('Type', 'config', 'Type'),
        ('Size', 'config', 'Size'),
        ('Dataset', 'config', 'Dataset')
    ]

    data = {
        col_name: [getattr(run, attr)[key] for run in runs]
        for col_name, attr, key in columns
    }

    return pd.DataFrame(data)