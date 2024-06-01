import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

scalability_ranking = ["comparison", "small", "large", "original"]
complexity_ranking = ["025", "030", "050", "075", "100"]


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
    # data2 = data[data['Scenario'] != "comparison"]
    filtered_data = data[data['Size'] == "100"]
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Scenario", y=y, order=scalability_ranking, hue="Dataset", hue_order=["PEMS-BAY", "METR-LA"], palette="dark", height=6
    )
    g.set_axis_labels("Map Area", y)
    g.fig.suptitle(y + " for both datasets")
    g.fig.subplots_adjust(top=0.9)
    ax = g.facet_axis(0, 0)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.2f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge')
    plt.savefig("figures/scalability-" + t + ".png", bbox_inches='tight')
    plt.show()

def plot_complexity(y, data, dataset, t):
    data2 = data[data['Scenario'].isin(["small", "large"])]
    filtered_data = data2[data2['Dataset'] == dataset]
    g = sns.catplot(
        data=filtered_data, kind="bar",
        x="Size", y=y, order=complexity_ranking, hue="Scenario", hue_order=["small", "large"], palette="dark", height=6
    )
    g.set_axis_labels("Map Area", y)
    g.fig.suptitle(y + " for " + dataset)
    g.fig.subplots_adjust(top=0.9)
    ax = g.facet_axis(0, 0)
    for i in range(0, len(ax.containers)):
        c = ax.containers[i]
        labels = [f"{val:.2f}" for val in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='edge')
    plt.savefig("figures/" + dataset + "/complexity-" + t + "-" + dataset.lower() + ".png", bbox_inches='tight')
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