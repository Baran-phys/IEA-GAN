"""
Module hosting various utility classes and functions
"""
import datetime
import json
import math
import os
import time
import pathlib

from collections import OrderedDict
from typing import List, Union

import boost_histogram as bh
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image

from tqdm import tqdm

from .norm import denorm


# Convenience dicts for activation
ACTIVATION_DICT = {
    "inplace_relu": torch.nn.ReLU(inplace=True),
    "relu": torch.nn.ReLU(inplace=False),
    "leaky_relu": torch.nn.LeakyReLU(0.2, inplace=False),
}


class Distribution(torch.Tensor):
    """
    A highly simplified convenience class for sampling from distributions
    One could also use PyTorch's inbuilt distributions package.
    Note that this class requires initialization to proceed as
    x = Distribution(torch.randn(size))
    x.init_distribution(dist_type, **dist_kwargs)
    x = x.to(device,dtype)
    This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
    """

    def init_distribution(self, dist_type: str, **kwargs):
        """
        Init the params of the distribution

        Args:
            dist_type (str): type of probability distribution
        """
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == "normal":
            self.mean, self.var = kwargs["mean"], kwargs["var"]
        elif self.dist_type == "categorical":
            self.num_categories = kwargs["num_categories"]
        elif self.dist_type == "censored_normal":
            self.mean, self.var = kwargs["mean"], kwargs["var"]
        elif self.dist_type == "bernoulli":
            pass
        elif self.dist_type=='truncated_normal':
            self.threshold = kwargs['threshold']
        elif self.dist_type=='permuted':
            self.num_categories = kwargs['num_categories']
        else:
            raise NotImplementedError(
                f"Distribution '{self.dist_type}' is not implemented"
            )

    def sample_(self):
        """
        Sample from distribution

        Raises:
            NotImplementedError: if dist_type is unknown.
        """
        if self.dist_type == "normal":
            self.normal_(self.mean, self.var)
        elif self.dist_type == "categorical":
            self.random_(0, self.num_categories)
        elif self.dist_type == "censored_normal":
            self.normal_(self.mean, self.var)
            self.relu_()
        elif self.dist_type == "bernoulli":
            self.bernoulli_()
        elif self.dist_type=='truncated_normal':
            raise NotImplementedError("truncated_normal is not defined. Please fix.")
            # v = truncated_normal(self.shape, self.threshold)
            # self.set_(v.float().cuda())
        elif self.dist_type=='permuted':
            permutations = torch.randperm(
                self.num_categories,
                device=self.dist_kwargs.get("device", "cuda")
            )
            if self.dist_kwargs.get("device", "cuda") == "cuda":
                self.set_(permutations.long().cuda())
            else:
                self.set_(permutations.long())
        else:
            raise NotImplementedError(
                f"Distribution '{self.dist_type}' is not implemented"
            )

    def to(self, *args, **kwargs):
        """
        Silly hack: overwrite the to() method to wrap the new object
        in a distribution as well
        """
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', fp16= False, z_var=1.0,
                z_dist='normal', threshold=1, y_dist='permuted' , ngd=False, fixed = False):
    if ngd:
        Tensor = torch.cuda.FloatTensor
        if fixed:
            z_ = Variable(Tensor(np.random.uniform(-1, 1, (G_batch_size, dim_z))), requires_grad=False)
        else:
            z_ = Variable(Tensor(np.random.uniform(-1, 1, (G_batch_size, dim_z))), requires_grad=True)

    else:
        z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
        if z_dist=='normal':
            z_.init_distribution(z_dist, mean=0, var=z_var)
        elif z_dist=='censored_normal':
            z_.init_distribution(z_dist, mean=0, var=z_var)
        elif z_dist=='bernoulli':
            z_.init_distribution(z_dist)
        elif z_dist=='truncated_normal':
            z_.init_distribution(z_dist, threshold=threshold)
        z_ = z_.to(device,torch.float16 if fp16 else torch.float32)


        if fp16:
            z_ = z_.half()

    if y_dist=='categorical':
        y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
        y_.init_distribution('categorical', num_categories=nclasses, device=device)
        y_ = y_.to(device, torch.int64)
    elif y_dist=='permuted':
        y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
        y_.init_distribution('permuted', num_categories=nclasses, device=device)
        y_ = y_.to(device, torch.int64)

    return z_, y_


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {
            label: 0 for label in self.labels_set
        }
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False
            )
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[
                    class_
                ] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


def seed_rng(seed: int):
    """
    Set rng seeds of torch, torch.cuda and numpy to a manual value.
    Args:
        seed (int): seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def join_strings(delimiter: str, strings: List[Union[str, None]]) -> str:
    """
    Function to join strings or ignore them if None
    Args:
        delimiter (str): string delimiter
        strings (List[Union[str, None]]): list of str or None

    Returns:
        str: joined string
    """
    return delimiter.join([item for item in strings if item])


def rename_weight_keys(state_dict: OrderedDict, fragment: str, replacement: str) -> OrderedDict:
    """
    Replace name fragments of state_dict keys.

    Args:
        state_dict (OrderedDict): torch state_dict
        fragment (str): fragment to be replaced
        replacement (str): substitution string

    Returns:
        OrderedDict: Returns new dictionary with modified keys.
    """
    state_dict_new = OrderedDict()

    for key, value in state_dict.items():
        state_dict_new[key.replace(fragment, replacement)] = value
    return state_dict_new


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


def make_mask(labels, n_cls, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels == c)
        mask_multi[c, c_indices] = +1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)


def initiate_standing_stats(net):
    for module in net.modules():
        if hasattr(module, "accumulate_standing"):
            module.reset_stats()
            module.accumulate_standing = True


def accumulate_standing_stats(net, z, y, nclasses, num_accumulations=16):
    initiate_standing_stats(net)
    net.train()
    for i in range(num_accumulations):
        with torch.no_grad():
            z.normal_()
            y.random_(0, nclasses)
            x = net(
                z, net.shared(y)
            )  # No need to parallelize here unless using syncbn
    # Set to eval mode
    net.eval()


def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config):
    # output path to samples directory
    rootpath = (
        pathlib.Path(config["outputroot"])
        .joinpath(config["run_name"])
        .joinpath("samples")
    )
    # Save an additional copy to mitigate accidental corruption if process is killed during a save
    save_weights(
        G,
        D,
        state_dict,
        config,
        "copy%d" % state_dict["itr"],
        G_ema if config["ema"] else None,
    )

    if config["num_save_copies"] > 0:
        # save_weights(G, D, state_dict, config['weights_root'],
        # experiment_name,
        #'copy%d' %  state_dict['save_num'],
        # G_ema if config['ema'] else None)
        state_dict["save_num"] = (state_dict["save_num"] + 1) % config[
            "num_save_copies"
        ]

    # Use EMA G for samples
    which_G = G_ema if config["ema"] and config["use_ema"] else G

    # Accumulate standing statistics?
    if config["accumulate_stats"]:
        accumulate_standing_stats(
            G_ema if config["ema"] and config["use_ema"] else G,
            z_,
            y_,
            config["n_classes"],
            config["num_standing_accumulations"],
        )

    # Save a random sample sheet with fixed z and y
    with torch.no_grad():
        imgs = which_G(fixed_z, fixed_y).float().cpu()
        imgs = denorm(imgs)

    # Path to fixed samples image output
    image_filename = rootpath.joinpath(f"fixed_samples{state_dict['itr']}.jpg").absolute()

    # There is a change here due to updated pytorch env
    save_image(
        torch.from_numpy(imgs.float().cpu().numpy()),
        image_filename,
        nrow=int(imgs.shape[0] ** 0.5),
        normalize=False,
    )

    sample_sheet(
        which_G,
        classes_per_sheet=40,
        num_classes=40,
        samples_per_class=10,
        config=config,
        folder_number=state_dict["itr"],
        z_=z_,
    )
    # save_image(imgs.data[:16], f'{GEN_IMGS_PATH}/step-{step:05}.png',
    # nrow=2, normalize=False)
    # plot_imgs(sample_images, size=2)


# fake histograms
def get_fake_stats_old(model, hist_axis, do_inverse_trf_hist=False, n=100):
    means = []
    occs = []
    # latents = torch.randn(100, 140, device=device)
    latents = trunc_trick(100, 140, bound=0.8)
    # labels =  torch.randint(0, 40, size=(8,), dtype=torch.long, device=device)
    hist = bh.Histogram(hist_axis)
    occ_hist = bh.Histogram(bh.axis.Regular(200, 0, 0.02))
    for i in tqdm(range(40), desc="Fake stats", total=40):
        # n = 100
        a = [i for _ in range(n)]
        labels = torch.LongTensor(a)
        imgs = model(latents, model.shared(labels)).detach().cpu()
        imgs = imgs[:, :, 3:-3, :]
        imgs = imgs.numpy()
        # imgs = F.interpolate(imgs,(250,768)).numpy()
        means.append(imgs[imgs > -0.25].mean())
        occs.append((imgs > -0.25).mean())
        occ_hist.fill((imgs > -0.25).mean(axis=(1, 2, 3)))
        if do_inverse_trf_hist:
            raise NotImplementedError("np_log_transform_inv_f is not defined. Please fix.")
            #hist.fill(np_log_transform_inv_f(imgs.ravel()))
        else:
            hist.fill(imgs.ravel())
    return hist, occ_hist, means, occs


# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device="cuda"):
    return torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# Sample function for sample sheets
def sample_sheet(
    G,
    classes_per_sheet,
    num_classes,
    samples_per_class,
    config,
    folder_number,
    z_=None,
):
    # Prepare sample directory
    rootpath = (
        pathlib.Path(config["outputroot"])
        .joinpath(config["run_name"])
        .joinpath("samples")
        .joinpath(f"{folder_number}")
    )
    device = config["device"]

    if not rootpath.exists():
        rootpath.mkdir()

    # loop over total number of sheets
    for i in range(num_classes // classes_per_sheet):
        ims = []
        y = torch.arange(
            i * classes_per_sheet, (i + 1) * classes_per_sheet, device=device
        )
        for j in range(samples_per_class):
            if (
                (z_ is not None)
                and hasattr(z_, "sample_")
                and classes_per_sheet <= z_.size(0)
            ):
                z_.sample_()
            else:
                z_ = torch.randn(classes_per_sheet, G.dim_z, device=device)
            with torch.no_grad():
                o = G(z_[:classes_per_sheet], y)
                o = denorm(o)

            ims += [o.data.cpu()]
        # This line should properly unroll the images
        out_ims = (
            torch.stack(ims, 1)
            .view(-1, ims[0].shape[1], ims[0].shape[2], ims[0].shape[3])
            .data.float()
            .cpu()
        )
        # added this line for updated pytorch env
        out_ims = torch.from_numpy(out_ims.numpy())
        # The path for the samples
        image_filename = rootpath.joinpath(
            f"samples{i}.jpg"
        ).absolute()

        torchvision.utils.save_image(
            out_ims, image_filename, nrow=samples_per_class, normalize=False
        )


# Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
def interp(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device="cuda").to(
        x0.dtype
    )
    return (x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1))


# interp sheet function
# Supports full, class-wise and intra-class interpolation
def interp_sheet(
    G,
    num_per_sheet,
    num_midpoints,
    num_classes,
    samples_root,
    experiment_name,
    folder_number,
    sheet_number=0,
    fix_z=False,
    fix_y=False,
    device="cuda",
):
    # Prepare zs and ys
    if fix_z:  # If fix Z, only sample 1 z per row
        zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
        zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
    else:
        zs = interp(
            torch.randn(num_per_sheet, 1, G.dim_z, device=device),
            torch.randn(num_per_sheet, 1, G.dim_z, device=device),
            num_midpoints,
        ).view(-1, G.dim_z)
    if fix_y:  # If fix y, only sample 1 z per row
        ys = sample_1hot(num_per_sheet, num_classes)
        ys = G.shared(ys).view(num_per_sheet, 1, -1)
        ys = ys.repeat(1, num_midpoints + 2, 1).view(
            num_per_sheet * (num_midpoints + 2), -1
        )
    else:
        ys = interp(
            G.shared(sample_1hot(num_per_sheet, num_classes)).view(
                num_per_sheet, 1, -1
            ),
            G.shared(sample_1hot(num_per_sheet, num_classes)).view(
                num_per_sheet, 1, -1
            ),
            num_midpoints,
        ).view(num_per_sheet * (num_midpoints + 2), -1)
    # Run the net--note that we've already passed y through G.shared.
    with torch.no_grad():
        out_ims = G(zs, G.shared(ys)).data.cpu()
        out_ims = denorm(out_ims)

    interp_style = (
        "" + ("Z" if not fix_z else "") + ("Y" if not fix_y else "")
    )
    image_filename = "%s/%s/%d/interp%s%d.jpg" % (
        samples_root,
        experiment_name,
        folder_number,
        interp_style,
        sheet_number,
    )
    torchvision.utils.save_image(
        out_ims, image_filename, nrow=num_midpoints + 2, normalize=False
    )


# Convenience debugging function to print out gradnorms and shape from each layer
def print_grad_norms(net):
    gradsums = [
        [
            float(torch.norm(param.grad).item()),
            float(torch.norm(param).item()),
            param.shape,
        ]
        for param in net.parameters()
    ]
    order = np.argsort([item[0] for item in gradsums])
    print(
        [
            "%3.3e,%3.3e, %s"
            % (
                gradsums[item_index][0],
                gradsums[item_index][1],
                str(gradsums[item_index][2]),
            )
            for item_index in order
        ]
    )


def get_singular_values(module: torch.nn.Module, prefix: str) -> dict:
    """
    Get singular values in order to log them.
    This will use the state dict to find them and substitute underscores for dots.

    Args:
        module (torch.nn.Module): torch neural net module
        prefix (str): string prefix for keys

    Returns:
        dict: mapping with state dict keys and singular values
    """
    state_dict = module.state_dict()
    return {
        f"{prefix}_{key}".replace(".", "_"): float(state_dict[key].item())
        for key in filter(lambda str_key: "sv" in str_key, state_dict)
    }


# Load a model's weights, optimizer, and the state_dict
def load_weights(
    G,
    D,
    state_dict,
    configuration: dict,
    weight_name=None,
    G_ema=None,
    strict=True,
    load_optim=True,
):
    # Path to weights folder
    rootpath = (
        pathlib.Path(configuration["outputroot"])
        .joinpath(configuration["run_name"])
        .joinpath("weights")
    )
    # Specify map location to be able to crossload into GPU or CPU
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if weight_name:
        print(f"Loading {weight_name} weights from {rootpath.absolute()}...")
    else:
        print(f"Loading weights from {rootpath.absolute()}...")
    if G is not None:
        filepath = rootpath.joinpath(f"{join_strings('_', ['G', weight_name])}.pth").absolute()
        weights_dict = torch.load(filepath, map_location=map_location)
        try:
            G.load_state_dict(weights_dict, strict=strict)
        except RuntimeError:
            # Quick fix to correct renaming of layers.
            # layer 'transG' was renamed to 'RR_G'
            # keys have to be renamed to import old weights
            print("Mismatch between file weight keys and model keys. Try renaming.")
            G.load_state_dict(rename_weight_keys(weights_dict, "transG", "RR_G"), strict=strict)
        if load_optim:
            filepath = rootpath.joinpath(
                f"{join_strings('_', ['G_optim', weight_name])}.pth"
            ).absolute()
            weights_dict = torch.load(filepath, map_location=map_location)
            try:
                G.optim.load_state_dict(weights_dict)
            except RuntimeError:
                print("Mismatch between file weight keys and model keys. Try renaming.")
                G.optim.load_state_dict(rename_weight_keys(weights_dict, "transG", "RR_G"))

    if D is not None:
        filepath = rootpath.joinpath(f"{join_strings('_', ['D', weight_name])}.pth").absolute()
        weights_dict = torch.load(filepath, map_location=map_location)
        try:
            D.load_state_dict(weights_dict, strict=strict)
        except RuntimeError:
            print("Mismatch between file weight keys and model keys. Try renaming.")
            D.load_state_dict(rename_weight_keys(weights_dict, "transcoder", "RR_D"), strict=strict)
        if load_optim:
            filepath = rootpath.joinpath(
                f"{join_strings('_', ['D_optim', weight_name])}.pth"
            ).absolute()
            weights_dict = torch.load(filepath, map_location=map_location)
            try:
                D.optim.load_state_dict(weights_dict)
            except RuntimeError:
                print("Mismatch between file weight keys and model keys. Try renaming.")
                D.optim.load_state_dict(rename_weight_keys(weights_dict, "transcoder", "RR_D"))
    # Load state dict
    for item in state_dict:
        state_dict[item] = torch.load(
            rootpath.joinpath(
                f"{join_strings('_', ['state_dict', weight_name])}.pth"
            ).absolute()
        )[item]
    if G_ema is not None:
        filepath = rootpath.joinpath(f"{join_strings('_', ['G_ema', weight_name])}.pth").absolute()
        weights_dict = torch.load(filepath, map_location=map_location)
        try:
            G_ema.load_state_dict(weights_dict, strict=strict)
        except RuntimeError:
            print("Mismatch between file weight keys and model keys. Try renaming.")
            G_ema.load_state_dict(rename_weight_keys(weights_dict, "transG", "RR_G"), strict=strict)


def write_metadata(configuration: dict, state_dict: dict):
    """Write some metadata to the logs directory

    Args:
        configuration (dict): run configuration
        state_dict (dict): state dictionary
    """
    metalogpath = (
        pathlib.Path(configuration["outputroot"])
        .joinpath(configuration["run_name"])
        .joinpath("logs")
        .joinpath("metalog.txt")
    )
    with open(metalogpath.absolute(), "w") as writefile:
        writefile.write("datetime: %s\n" % str(datetime.datetime.now()))
        writefile.write("state: %s\n" % str(state_dict))


def save_weights(G, D, state_dict, configuration, name_suffix=None, G_ema=None):
    """
    Save a model's weights, optimizer, and the state_dict
    """
    weightspath = (
        pathlib.Path(configuration["outputroot"])
        .joinpath(configuration["run_name"])
        .joinpath("weights")
    )
    if name_suffix:
        print("Saving weights to %s/%s..." % (weightspath.absolute(), name_suffix))
    else:
        print("Saving weights to %s..." % weightspath.absolute())
    torch.save(
        G.state_dict(),
        "%s/%s.pth" % (weightspath.absolute(), join_strings("_", ["G", name_suffix])),
    )
    torch.save(
        G.optim.state_dict(),
        "%s/%s.pth" % (weightspath.absolute(), join_strings("_", ["G_optim", name_suffix])),
    )
    torch.save(
        D.state_dict(),
        "%s/%s.pth" % (weightspath.absolute(), join_strings("_", ["D", name_suffix])),
    )
    torch.save(
        D.optim.state_dict(),
        "%s/%s.pth" % (weightspath.absolute(), join_strings("_", ["D_optim", name_suffix])),
    )
    torch.save(
        state_dict,
        "%s/%s.pth" % (weightspath.absolute(), join_strings("_", ["state_dict", name_suffix])),
    )
    if G_ema is not None:
        torch.save(
            G_ema.state_dict(),
            "%s/%s.pth" % (weightspath.absolute(), join_strings("_", ["G_ema", name_suffix])),
        )


class Adam16(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

    # Safety modification to make sure we floatify our state
    def load_state_dict(self, state_dict):
        super(Adam16, self).load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["exp_avg"] = self.state[p]["exp_avg"].float()
                self.state[p]["exp_avg_sq"] = self.state[p][
                    "exp_avg_sq"
                ].float()
                self.state[p]["fp32_p"] = self.state[p]["fp32_p"].float()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.new().resize_as_(grad).zero_()
                    # Fp32 copy of the weights
                    state["fp32_p"] = p.data.float()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], state["fp32_p"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"]
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                state["fp32_p"].addcdiv_(-step_size, exp_avg, denom)
                p.data = state["fp32_p"].half()

        return loss


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class apply_ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print("Initializing EMA parameters to be source parameters...")
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(
                    self.target_dict[key].data * decay
                    + self.source_dict[key].data * (1 - decay)
                )


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=None):
    if blacklist is None:
        blacklist = []
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if len(param.shape) < 2 or any(
                [param is item for item in blacklist]
            ):
                continue
            w = param.view(param.shape[0], -1)
            grad = 2 * torch.mm(
                torch.mm(w, w.t())
                * (1.0 - torch.eye(w.shape[0], device=w.device)),
                w,
            )
            param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=None):
    if blacklist is None:
        blacklist = []
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = 2 * torch.mm(
                torch.mm(w, w.t()) - torch.eye(w.shape[0], device=w.device), w
            )
            param.grad.data += strength * grad.view(param.shape)


def trunc_trick(bs, z_dim, bound=1):
    z = torch.randn(bs, z_dim)
    while z.min() < -bound or bound < z.max():
        z = z.where((-bound < z) & (z < bound), torch.randn_like(z))
    return z


def collect_bn_stats(G, n_samples, config):
    im_batch_size = config["batch_size"]
    device = config["device"]
    G.train()

    for i_batch in range(0, n_samples, im_batch_size):
        with torch.no_grad():
            z = torch.randn(im_batch_size, G.dim_z, device=device)
            y = torch.arange(im_batch_size).to(device)
            images = G(z, G(y)).float().cpu()


def generate_images(out_dir, G, n_images, config):
    im_batch_size = config["batch_size"]
    z_bound = config["trunc_z"]
    device = config["device"]
    if z_bound > 0.0:
        print(f"Truncating z to (-{z_bound}, {z_bound})")

    for i_batch in range(0, n_images, im_batch_size):
        with torch.no_grad():
            if z_bound > 0.0:
                z = trunc_trick(im_batch_size, G.dim_z, bound=z_bound).to(
                    device
                )
            else:
                z = torch.randn(im_batch_size, G.dim_z, device=device)
            y = torch.arange(im_batch_size).to(device)
            images = G(z, G(y)).float().cpu()

        if i_batch + im_batch_size > n_images:
            n_last_images = n_images - i_batch
            print(
                f"Taking only {n_last_images} images from the last batch..."
            )
            images = images[:n_last_images]

        for i_image, image in enumerate(images):
            fname = os.path.join(out_dir, f"image_{i_batch+i_image:05d}.png")
            image = denorm(image)
            if config["denoise"]:
                image = image * 256
                image = image.numpy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.fastNlMeansDenoisingColored(
                    src=image,
                    dst=None,
                    h=config["denoise_str_lum"],
                    hColor=config["denoise_str_col"],
                    templateWindowSize=config["denoise_kernel_size"],
                    searchWindowSize=config["denoise_search_window"],
                )

                cv2.imwrite(fname, image)
            else:
                torchvision.utils.save_image(image, fname)


def reshape_weight_to_matrix(weight):
    weight_mat = weight
    dim = 0
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(
            dim, *[d for d in range(weight_mat.dim()) if d != dim]
        )
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i


def calculate_sv(model):
    sigmas = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (
                "weight" in name
                and "bn" not in name
                and "shared" not in name
                and "deconv" not in name
            ):
                if "blocks" in name:
                    splited_name = name.split(".")
                    idx = find_string(splited_name, "blocks")
                    block_idx = int(splited_name[int(idx + 1)])
                    module_idx = int(splited_name[int(idx + 2)])
                    operation_name = splited_name[idx + 3]
                    operations = model.blocks[block_idx][module_idx]
                    operation = getattr(operations, operation_name)
                else:
                    splited_name = name.split(".")
                    idx = -1
                    operation_name = splited_name[idx + 1]
                    operation = getattr(model, operation_name)

                weight_orig = reshape_weight_to_matrix(operation.weight_orig)
                weight_u = operation.weight_u
                weight_v = operation.weight_v
                sigmas[name] = torch.dot(
                    weight_u, torch.mv(weight_orig, weight_v)
                )
    return sigmas
