#!/usr/bin/env python

import argparse
import pickle
import pathlib
import sys

import torch
import boost_histogram as bh
from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf

from dataset import dataset_from_dir

# Add IEA-GAN to path so that model can be imported
sys.path.append(f"{pathlib.Path(__file__).parent.parent.absolute()}")

import model  # pylint: disable=import-error,wrong-import-position


device = "cuda" if torch.cuda.is_available() else "cpu"


IEAGAN_CONFIG = {
    'skip_init': True, 'no_optim': True, 'resolution': 256,
    'D_attn' : '0', 'G_attn':'0', 'attn_type':'sa','n_head_G':2,
    'G_ch' : 32, 'D_ch' : 32, 'relational_embed' : True, 'rdof_dim': 4,
    'dim_z': 128, 'H_base': 3, 'G_shared': True, 'device': device,
    'shared_dim': 128, 'hier': True, 'prior_embed': False
}


THRESHOLD = 7


def load_model(model, state_path, device=device):
    state = None
    try:
        state = torch.load(state_path, map_location=device)
        model.to(device)
        model.load_state_dict(state)
    finally:
        del state
    return model


def load_all_models():
    return {
        "IEAGAN": load_model(
            model.Generator(**IEAGAN_CONFIG).to(device, torch.float32),
            "weights/IEAGAN.pth"
        ),
        "ContraGAN": load_model(
            model.Generator(
                **dict(IEAGAN_CONFIG, relational_embed=False, rdof_dim= 0)
            ).to(device, torch.float32),
            "weights/ContraGAN.pth"
        ),
        "PEGAN": load_model(
            model.Generator(
                **dict(IEAGAN_CONFIG, G_attn= '32', relational_embed=False, prior_embed=True, rdof_dim= 0)
            ).to(device, torch.float32),
            "weights/PEGAN.pth"
        ),
        "BigGAN_deep": load_model(
            model.Generator(
                **dict(IEAGAN_CONFIG, relational_embed=False, prior_embed=False, rdof_dim= 0)
            ).to(device, torch.float32),
            "weights/BigGAN_deep.pth"
        ),
    }


def get_stats(
    ds,
    hist_axis=bh.axis.Variable([-1, 1, 7] + list(np.linspace(8, 256, 249))),
    hist_axis_occ=bh.axis.Regular(200, 0, 0.02),
    n_events=100,
    desc=None
):
    means = []
    occs = []
    mean_act = []
    hist = bh.Histogram(hist_axis)
    occ_hist = bh.Histogram(hist_axis_occ)
    for i, batch in tqdm(zip(range(n_events), ds), desc=desc, total=n_events):
        imgs, labels = batch
        mask = imgs > 0 # threshold cut should already be applied
        occs.append(tf.reduce_mean(tf.cast(mask, dtype=tf.float32), axis=(1, 2, 3)))
        means.append(
            (
                tf.reduce_sum(tf.where(mask, imgs, 0), axis=(1, 2, 3))
                / tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=(1, 2, 3))
            ).numpy()
        )
        hist.fill(np.asarray(imgs).ravel())
        occ_hist.fill(np.asarray(mask).mean(axis=(1, 2, 3)))
    means = np.array(means).mean(axis=0)
    occs = np.array(occs).mean(axis=0)
    return hist, occ_hist, means, occs


def log_transform_inv(img):
    img = 0.5 * (img + 1)
    return (np.exp(np.log(256.0) * img) - 1)


def generate_images(model):
    while True:
        with torch.no_grad():
            latents = torch.randn(40, 128, device=device)
            labels = torch.tensor([c for c in range(40)], dtype=torch.long, device=device)
            imgs = log_transform_inv(model(latents, labels).cpu().numpy())
            imgs[imgs < THRESHOLD] = 0
            imgs = imgs[:,:,3:-3,:] # crop
            yield (
                imgs,
                labels.cpu().numpy()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir")
    args = parser.parse_args()

    ds, label_names = dataset_from_dir(
        pathlib.Path(args.datadir).absolute(), return_labelnames=True, zip_labels=True, shuffle=False, do_log_transform=False
    )
    N_EVENTS = 10000

    real_stats = get_stats(ds.batch(40), n_events=N_EVENTS, desc="Real stats")

    models = load_all_models()
    fake_stats = {}
    for name, model in models.items():
        fake_stats[name] = get_stats(generate_images(model), n_events=N_EVENTS, desc=name)

    all_stats = {"real": real_stats}
    all_stats.update(fake_stats)

    with open("eval_results_10k.pickle", "wb") as f:
        pickle.dump(all_stats, f)
