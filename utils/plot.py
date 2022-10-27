"""Module for all plotting and visualization functions"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from .norm import denorm


def plot_imgs(imgs, cols=2, size=8):
    n = imgs.shape[0]
    rows = n // cols
    _, axes = plt.subplots(
        figsize=(cols * size, rows * size), ncols=cols, nrows=rows
    )
    for i, ax in enumerate(axes.flatten()):
        img = denorm(imgs[i]).permute(1, 2, 0)
        img = torch.squeeze(img).numpy()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img, cmap="gray", interpolation="none")
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()


def plot_sim_heatmap(similarity, xlabels, ylabels, mode, configuration, state_dict):
    sns.set(style="white")
    fig, axis = plt.subplots(figsize=(18, 18))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(similarity, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        similarity,
        mask=mask,
        cmap=cmap,
        center=0.5,
        xticklabels=xlabels,
        yticklabels=ylabels,
        square=True,
        linewidths=0.5,
        fmt=".2f",
        annot=True,
        cbar_kws={"shrink": 0.5},
        vmax=1,
        annot_kws={"size": 8},
    )

    axis.set_title("Heatmap of cosine similarity scores").set_fontsize(15)
    axis.set_xlabel("")
    axis.set_ylabel("")
    # fig.ioff()
    plt.close(fig)
    # Path to samples folder
    samplepath = (
        pathlib.Path(configuration["outputroot"])
        .joinpath(configuration["run_name"])
        .joinpath("samples")
    )
    # Create filename with mode and iteration
    savepath = samplepath.joinpath(
        f"{mode}_sim_heatmap{state_dict['itr']}.png"
    )
    # Save figure
    fig.savefig(savepath.absolute())
