import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean plotting style without grid
sns.set_theme(style="white")


def plot_random_cost():
    # Specify the path to the JSON file
    file_path = '../logs/log_random_cost.json'

    try:
        # Load data from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract the list of experiments and create a pandas DataFrame
        df = pd.DataFrame(data['experiments'])

        # Filter the DataFrame based on the k values range
        df_filtered = df[(df['k'] >= 32) & (df['k'] <= 1294)].copy()

        # Convert columns to numeric types
        df_filtered['k'] = pd.to_numeric(df_filtered['k'], errors='coerce')
        df_filtered['diffusion_ratio'] = pd.to_numeric(df_filtered['diffusion_ratio'], errors='coerce')

        # Drop rows with invalid values
        df_filtered.dropna(subset=['k', 'diffusion_ratio'], inplace=True)

        # Sort data by k to ensure the line plot connects points correctly
        df_filtered.sort_values(by="k", inplace=True)

        # Get the list of unique algorithms
        algorithms = df_filtered['algorithm'].unique()

        # Iterate through each algorithm to create a plot
        for algorithm in algorithms:
            if algorithm == 'greedy_seed_set':
                df_algo = df_filtered[df_filtered['algorithm'] == algorithm]
                heuristics = df_algo['euristic'].unique()

                for heuristic in heuristics:
                    df_heuristic = df_algo[df_algo['euristic'] == heuristic].copy()
                    df_heuristic.sort_values(by="k", inplace=True)

                    plt.figure(figsize=(12, 7))

                    # Draw the connecting line first
                    sns.lineplot(
                        data=df_heuristic, x='k', y='diffusion_ratio',
                        color="#1f77b4", linewidth=2.5, zorder=1
                    )

                    # Draw larger markers on top of the line
                    sns.scatterplot(
                        data=df_heuristic, x='k', y='diffusion_ratio',
                        s=80, marker="o", color="#1f77b4", edgecolor="black", zorder=2
                    )

                    # Add labels above each point showing the diffusion ratio
                    for _, row in df_heuristic.iterrows():
                        plt.text(
                            row['k'], row['diffusion_ratio'] + 0.02,
                            f"{row['diffusion_ratio']:.2f}",
                            ha="center", va="bottom", fontsize=9, color="black"
                        )

                    # Set titles and labels
                    plt.title(f'Algorithm: {algorithm} - Euristic: {heuristic.upper()}', fontsize=16)
                    plt.xlabel('K Values', fontsize=12)
                    plt.ylabel('Diffusion Ratio', fontsize=12)
                    plt.xlim(0, 1350)
                    plt.ylim(0, 1)
                    plt.show()

            else:
                df_algo = df_filtered[df_filtered['algorithm'] == algorithm].copy()
                df_algo.sort_values(by="k", inplace=True)

                plt.figure(figsize=(12, 7))

                # Draw the connecting line first
                sns.lineplot(
                    data=df_algo, x='k', y='diffusion_ratio',
                    color="#1f77b4", linewidth=2.5, zorder=1
                )

                # Draw larger markers on top of the line
                sns.scatterplot(
                    data=df_algo, x='k', y='diffusion_ratio',
                    s=80, marker="o", color="#1f77b4", edgecolor="black", zorder=2
                )

                # Add labels above each point
                for _, row in df_algo.iterrows():
                    plt.text(
                        row['k'], row['diffusion_ratio'] + 0.02,
                        f"{row['diffusion_ratio']:.2f}",
                        ha="center", va="bottom", fontsize=9, color="black"
                    )

                # Set titles and labels
                plt.title(f'Algorithm: {algorithm}', fontsize=16)
                plt.xlabel('K Values', fontsize=12)
                plt.ylabel('Diffusion Ratio', fontsize=12)
                plt.xlim(0, 1350)
                plt.ylim(0, 1)
                plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def plot_half_node_degree_cost():
    # Specify the path to the JSON file
    file_path = '../logs/log_half_node_degree_cost.json'

    try:
        # Load data from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract the list of experiments and create a pandas DataFrame
        df = pd.DataFrame(data['experiments'])

        # Filter the DataFrame based on the k values range
        df_filtered = df[(df['k'] >= 69) & (df['k'] <= 2779)].copy()

        # Convert columns to numeric types
        df_filtered['k'] = pd.to_numeric(df_filtered['k'], errors='coerce')
        df_filtered['diffusion_ratio'] = pd.to_numeric(df_filtered['diffusion_ratio'], errors='coerce')

        # Drop rows with invalid values
        df_filtered.dropna(subset=['k', 'diffusion_ratio'], inplace=True)

        # Sort data by k to ensure the line plot connects points correctly
        df_filtered.sort_values(by="k", inplace=True)

        # Get the list of unique algorithms
        algorithms = df_filtered['algorithm'].unique()

        # Iterate through each algorithm to create a plot
        for algorithm in algorithms:
            if algorithm == 'greedy_seed_set':
                df_algo = df_filtered[df_filtered['algorithm'] == algorithm]
                heuristics = df_algo['euristic'].unique()

                for heuristic in heuristics:
                    df_heuristic = df_algo[df_algo['euristic'] == heuristic].copy()
                    df_heuristic.sort_values(by="k", inplace=True)

                    plt.figure(figsize=(12, 7))

                    # Draw the connecting line first
                    sns.lineplot(
                        data=df_heuristic, x='k', y='diffusion_ratio',
                        color="#1f77b4", linewidth=2.5, zorder=1
                    )

                    # Draw larger markers on top of the line
                    sns.scatterplot(
                        data=df_heuristic, x='k', y='diffusion_ratio',
                        s=80, marker="o", color="#1f77b4", edgecolor="black", zorder=2
                    )

                    # Add labels above each point showing the diffusion ratio
                    for _, row in df_heuristic.iterrows():
                        plt.text(
                            row['k'], row['diffusion_ratio'] + 0.02,
                            f"{row['diffusion_ratio']:.2f}",
                            ha="center", va="bottom", fontsize=9, color="black"
                        )

                    # Set titles and labels
                    plt.title(f'Algorithm: {algorithm} - Euristic: {heuristic.upper()}', fontsize=16)
                    plt.xlabel('K Values', fontsize=12)
                    plt.ylabel('Diffusion Ratio', fontsize=12)
                    plt.xlim(0, 2850)
                    plt.ylim(0, 1)
                    plt.show()

            else:
                df_algo = df_filtered[df_filtered['algorithm'] == algorithm].copy()
                df_algo.sort_values(by="k", inplace=True)

                plt.figure(figsize=(12, 7))

                # Draw the connecting line first
                sns.lineplot(
                    data=df_algo, x='k', y='diffusion_ratio',
                    color="#1f77b4", linewidth=2.5, zorder=1
                )

                # Draw larger markers on top of the line
                sns.scatterplot(
                    data=df_algo, x='k', y='diffusion_ratio',
                    s=80, marker="o", color="#1f77b4", edgecolor="black", zorder=2
                )

                # Add labels above each point
                for _, row in df_algo.iterrows():
                    plt.text(
                        row['k'], row['diffusion_ratio'] + 0.02,
                        f"{row['diffusion_ratio']:.2f}",
                        ha="center", va="bottom", fontsize=9, color="black"
                    )

                # Set titles and labels
                plt.title(f'Algorithm: {algorithm}', fontsize=16)
                plt.xlabel('K Values', fontsize=12)
                plt.ylabel('Diffusion Ratio', fontsize=12)
                plt.xlim(0, 2850)
                plt.ylim(0, 1)
                plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def plot_cost_bridge_capped():
    # Specify the path to the JSON file
    file_path = '../logs/log_cost_bridge_capped.json'

    try:
        # Load data from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract the list of experiments and create a pandas DataFrame
        df = pd.DataFrame(data['experiments'])

        # Filter the DataFrame based on the k values range
        df_filtered = df[(df['k'] >= 147) & (df['k'] <= 5914)].copy()

        # Convert columns to numeric types
        df_filtered['k'] = pd.to_numeric(df_filtered['k'], errors='coerce')
        df_filtered['diffusion_ratio'] = pd.to_numeric(df_filtered['diffusion_ratio'], errors='coerce')

        # Drop rows with invalid values
        df_filtered.dropna(subset=['k', 'diffusion_ratio'], inplace=True)

        # Sort data by k to ensure the line plot connects points correctly
        df_filtered.sort_values(by="k", inplace=True)

        # Get the list of unique algorithms
        algorithms = df_filtered['algorithm'].unique()

        # Iterate through each algorithm to create a plot
        for algorithm in algorithms:
            if algorithm == 'greedy_seed_set':
                df_algo = df_filtered[df_filtered['algorithm'] == algorithm]
                heuristics = df_algo['euristic'].unique()

                for heuristic in heuristics:
                    df_heuristic = df_algo[df_algo['euristic'] == heuristic].copy()
                    df_heuristic.sort_values(by="k", inplace=True)

                    plt.figure(figsize=(12, 7))

                    # Draw the connecting line first
                    sns.lineplot(
                        data=df_heuristic, x='k', y='diffusion_ratio',
                        color="#1f77b4", linewidth=2.5, zorder=1
                    )

                    # Draw larger markers on top of the line
                    sns.scatterplot(
                        data=df_heuristic, x='k', y='diffusion_ratio',
                        s=80, marker="o", color="#1f77b4", edgecolor="black", zorder=2
                    )

                    # Add labels above each point showing the diffusion ratio
                    for _, row in df_heuristic.iterrows():
                        plt.text(
                            row['k'], row['diffusion_ratio'] + 0.02,
                            f"{row['diffusion_ratio']:.2f}",
                            ha="center", va="bottom", fontsize=9, color="black"
                        )

                    # Set titles and labels
                    plt.title(f'Algorithm: {algorithm} - Euristic: {heuristic.upper()}', fontsize=16)
                    plt.xlabel('K Values', fontsize=12)
                    plt.ylabel('Diffusion Ratio', fontsize=12)
                    plt.xlim(0, 6000)
                    plt.ylim(0, 1)
                    plt.show()

            else:
                df_algo = df_filtered[df_filtered['algorithm'] == algorithm].copy()
                df_algo.sort_values(by="k", inplace=True)

                plt.figure(figsize=(12, 7))

                # Draw the connecting line first
                sns.lineplot(
                    data=df_algo, x='k', y='diffusion_ratio',
                    color="#1f77b4", linewidth=2.5, zorder=1
                )

                # Draw larger markers on top of the line
                sns.scatterplot(
                    data=df_algo, x='k', y='diffusion_ratio',
                    s=80, marker="o", color="#1f77b4", edgecolor="black", zorder=2
                )

                # Add labels above each point
                for _, row in df_algo.iterrows():
                    plt.text(
                        row['k'], row['diffusion_ratio'] + 0.02,
                        f"{row['diffusion_ratio']:.2f}",
                        ha="center", va="bottom", fontsize=9, color="black"
                    )

                # Set titles and labels
                plt.title(f'Algorithm: {algorithm}', fontsize=16)
                plt.xlabel('K Values', fontsize=12)
                plt.ylabel('Diffusion Ratio', fontsize=12)
                plt.xlim(0, 6000)
                plt.ylim(0, 1)
                plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


if __name__ == '__main__':
    plot_half_node_degree_cost()