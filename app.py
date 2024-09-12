import os
from  dotenv import load_dotenv
import gradio as gr
import pandas as pd
from OptimizeClusters import OptimizeClusters
from GenerateRunTypes import GenerateRunTypes
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

load_dotenv()

run_attributes = [
    'Distance', 'Calories', 'Time',
    'Avg HR', 'Max HR', 'Aerobic TE',
    'Avg Run Cadence', 'Max Run Cadence',
    'Avg Pace', 'Best Pace', 'Total Ascent',
    'Total Descent', 'Avg Stride Length',
    'Min Temp', 'Number of Laps', 'Max Temp',
    'Moving Time', 'Elapsed Time',
    'Min Elevation', 'Max Elevation'
]

def run_category_fig(X, y, run_attrs):
    fig = plt.figure(figsize=(14, 10))
    if len(run_attrs) == 2:
        ax = fig.add_subplot(111)
    elif len(run_attrs) == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError('Number of features should be 2 or 3')

    for run_category in y.unique():
        if len(run_attrs) == 2:
            ax.scatter(
                X.loc[y == run_category, run_attrs[0]],
                X.loc[y == run_category, run_attrs[1]],
                label=f'{run_category}'
            )
            ax.set_xlabel(run_attrs[0])
            ax.set_ylabel(run_attrs[1])
        elif len(run_attrs) == 3:
            ax.scatter(
                X.loc[y == run_category, run_attrs[0]],
                X.loc[y == run_category, run_attrs[1]],
                X.loc[y == run_category, run_attrs[2]],
                label=f'{run_category}'
            )

            ax.set_xlabel(run_attrs[0])
            ax.set_ylabel(run_attrs[1])
            ax.set_zlabel(run_attrs[2])

    ax.legend()

    return fig

def generate_n_clusters(filename, run_attrs, n_clusters=5):
    data = pd.read_csv(filename, header=0, sep=',')
    print(data.columns)
    X, y = SMOTE().fit_resample(
        data.drop(columns=['Date', 'Run Category', 'Run Type']),
        data['Run Category']
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )
    kmeans.fit(X[run_attrs])
    labels = kmeans.labels_
    print('LABELS: ', labels)

    fig = plt.figure(figsize=(14, 10))

    if len(run_attrs) == 2:
        ax = fig.add_subplot(111)
    elif len(run_attrs) == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError('Number of features should be 2 or 3')

    for cluster in range(n_clusters):
        if len(run_attrs) == 2:
            ax.scatter(
                X.loc[labels == cluster, run_attrs[0]],
                X.loc[labels == cluster, run_attrs[1]],
                label=f'Run Type {cluster + 1}'
            )
            ax.set_xlabel(run_attrs[0])
            ax.set_ylabel(run_attrs[1])
        elif len(run_attrs) == 3:
            ax.scatter(
                X.loc[labels == cluster, run_attrs[0]],
                X.loc[labels == cluster, run_attrs[1]],
                X.loc[labels == cluster, run_attrs[2]],
                label=f'Run Type {cluster + 1}'
            )

            ax.set_xlabel(run_attrs[0])
            ax.set_ylabel(run_attrs[1])
            ax.set_zlabel(run_attrs[2])

    ax.legend()

    return [run_category_fig(X, y, run_attrs), fig]


def generate_clusters(filename, n_clusters=5):
    data = pd.read_csv(filename, header=0, sep=',')
    pipeline = Pipeline(
    steps=[
            ('scaler', StandardScaler()),
            ('PCA', PCA(n_components=2)),
            ('Optimize K', OptimizeClusters())
        ]
    )

    result, cluster_centers, n_clusters = pipeline.fit_transform(
        data.drop(columns=['Date'])
    )

    clustered_data = data.drop(columns=['Date'])
    clustered_data['cluster'] = result[:, 2]
    avg_str = clustered_data.groupby('cluster').mean().to_csv(header=True)

    run_categories = GenerateRunTypes().generate(n_clusters, avg_str)
    print(run_categories)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        result[:, 0],
        result[:, 1],
        c=result[:, 2],
        cmap='viridis', s=50
    )

    for idx, center in enumerate(cluster_centers):
        plt.text(
            center[0], center[1],
            run_categories[idx],
            fontsize=8,
            ha='center',
            va='center',
            color='black',
            bbox=dict(facecolor='white', alpha=0.4)
        )

    return fig

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# Running Coach AI")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## Steps
                1. Take a look at the template file for the type of data required
                2. Upload your running data
                3. Select the number of run categories
                4. Click on Analyze Runs
                5. View the run categories
                6. Work with the coach to design a plan
                """
            )

        with gr.Column():
            download_button = gr.File(
                "./data/template.csv",
                label="Running Data Template"
            )

            data_file = gr.File(label="Running Data")
            n_clusters = gr.Slider(
                value=5, minimum=1, maximum=10, step=1, label="Number of Run Types"
            )
            run_attrs = gr.Dropdown(
                run_attributes,
                label="Select 2-3 Attributes",
                multiselect=True,
                render=True
            )
            analyze_runs_button = gr.Button("Analyze Runs")

    with gr.Row():
        with gr.Column():
            run_types_plot = gr.Plot(
                label="Logged Run Types",
                format="png",
                container=True
            )

        with gr.Column():
            run_cat_plot = gr.Plot(
                label="Clustered Run Types",
                format="png",
                container=True,

            )

    analyze_runs_button.click(
        generate_n_clusters,
        inputs=[data_file, run_attrs, n_clusters],
        outputs=[run_types_plot, run_cat_plot]
    )

if __name__ == "__main__":
    demo.launch()