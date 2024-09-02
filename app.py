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

load_dotenv()

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
                value=5, minimum=1, maximum=10, step=1, label="Run Categories"
            )
            analyze_runs_button = gr.Button("Analyze Runs")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Run Type Categories")
            run_cat_plot = gr.Plot(
                label="Run Categories",
                format="png",
                container=True
            )

    analyze_runs_button.click(
        generate_clusters,
        inputs=[data_file, n_clusters],
        outputs=run_cat_plot
    )
        # gr.Interface(
        #     fn=generate_clusters,
        #     inputs=[
        #         gr.File(label="Running Data"),
        #         gr.Slider(
        #             value=5, minimum=1, maximum=10, step=1, label="Number of Clusters"),
        #         gr.Slider(
        #             value=2, minimum=1, maximum=10, step=1, label="Number of Components"),
        #     ],
        #     outputs=gr.Plot(label="Run Categories", format="png")
        # )

    # import random
    # def random_response(message, history):
    #     return random.choice(["Yes", "No"])

    # with gr.Row():
    #     with gr.Column():
    #         gr.Markdown("Ask your coach for help...")
    #         gr.ChatInterface(
    #             random_response,
    #             examples=[
    #                 "Write me a running plan for 5k tailored to my stats",
    #                 "hola",
    #                 "merhaba"
    #             ]
    #         )

if __name__ == "__main__":
    demo.launch()