import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


def visualize_cvrp_scatter(data: dict, solution: dict = None, data_nm: str = None) -> None:
    """Plot CVRP solution routes based on weights, coordinates and routes"""
    # Define depot coordinates and remove it from all nodes coordinates
    depot_coords = data['coords'][1]
    cstmr_coords = np.delete(list(data['coords'].values()), 0, axis=0)
    cstmr_demand = np.delete(list(data['demands']), 0)
    
    # Create node ID to index mapping (excluding depot)
    node_ids = np.delete(np.arange(len(data['coords'])), 0)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Plot customer coordinates scatter plot with demand values
    scatter = sns.scatterplot(
        x=cstmr_coords[:, 0],
        y=cstmr_coords[:, 1],
        hue=cstmr_demand,
        s=50,
        palette=cmap,
        edgecolor='black',
        linewidth=0.55,
        alpha=0.5,
        ax=ax,
        legend=False
    )

    # Plot routes as lines connecting nodes
    solution_summary = ""
    if solution is not None:
        route_colors = sns.color_palette(
            "tab10", n_colors=len(solution['routes']))
        for route_idx, route in enumerate(solution['routes']):
            route_coords = [depot_coords]
            for node_id in route:
                idx = node_to_idx[node_id]
                route_coords.append(cstmr_coords[idx])
            route_coords.append(depot_coords)
            route_coords = np.array(route_coords)
            
            # Plot route line
            ax.plot(
                route_coords[:, 0],
                route_coords[:, 1],
                color=route_colors[route_idx % len(route_colors)],
                linewidth=1.5,
                alpha=0.7,
                label=f'Route {route_idx + 1}'
            )
            solution_summary = f" Solution cost: {format(round(solution['cost'], 1), ',')}."

    # Add depot (via default matplotlib.pyplot)
    ax.scatter(
        x=depot_coords[0],
        y=depot_coords[1],
        s=200,
        c='yellow',
        marker='*',
        edgecolors='darkorange',
        linewidths=1,
        label='Depot',
        zorder=3
    )

    # Add demand colorbar
    norm = plt.Normalize(vmin=cstmr_demand.min(), vmax=cstmr_demand.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Demand', fontsize=10)

    # Set-up axis
    ax.legend(loc="best", fontsize=8)
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.grid(axis="both", alpha=0.25)
    ax.set_title(f"{data_nm}." + solution_summary, loc="left")

    plt.tight_layout()
    plt.show()
