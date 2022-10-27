# pylint: skip-file
import os
import random
from tqdm import tqdm, trange
from glob import glob
import numpy as np
from PIL import Image
from scipy import linalg
import urllib.request
import requests
import shutil
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision
import contextlib

import cleanfid

# from utils import *    (instead copied here)
# from features import * (instead copied them here)
# from resize import *   (instead copied them here and edited the sigle channel func)
from cleanfid.downloads_helper import *
from cleanfid.inception_pytorch import InceptionV3

# from inception_torchscript import InceptionV3W (instead copied them here and edited the path to the new inceptionv3w)


#####inceptionv3pxd#####
class InceptionV3pxd(nn.Module):
    """
    Wrapper around Inception V3 torchscript model trained over the pxd dataset with +0.99 accuracy
    path: locally saved inception weights
    """

    def __init__(self, resize_inside=False):
        super(InceptionV3pxd, self).__init__()
        # use the current directory by default
        # path = os.path.join('/project/agkuhr/users/hosein47/pxdgen/model/analysis/con_gan/model/
        # BigGAN/dev_model/mycleanfid',
        #'inception_V3_best.pt')
        path = os.path.join("./mycleanfid/inception_V3_best.pt")
        self.base = torch.load(path, map_location="cpu").eval()
        self.resize_inside = resize_inside
        self.Apool = nn.AdaptiveAvgPool2d(1)

    """
    Get the inception features without resizing
    x: Image with values in range [0,255]
    """

    def forward(self, x):
        bs = x.shape[0]
        # make sure it is resized already
        assert (x.shape[2] == 299) and (x.shape[3] == 299)
        # apply normalization
        # x1 = x - 128
        # x2 = x1 / 128
        out = self.base.forward_features(x)
        features = self.Apool(out).view((bs, 2048))
        return features


####inceptionv3pxd End#######


####inceptionv3W#####
@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    # On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run. See
    #   https://github.com/GaParmar/clean-fid/issues/5
    #   https://github.com/pytorch/pytorch/issues/64062
    if torch.__version__.startswith("1.9."):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith("1.9."):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


class InceptionV3W(nn.Module):
    """
    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
    path: locally saved inception weights
    """

    def __init__(self, path, download=True, resize_inside=False):
        super(InceptionV3W, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        if download:
            check_download_inception(fpath=path)
        path = os.path.join(path, "inception-2015-12-05.pt")
        # path = os.path.join('/project/agkuhr/users/hosein47/pxdgen/model/analysis/con_gan/model/BigGAN/dev_model/mycleanfid',
        #'inception_V3_best.pt')
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers
        self.resize_inside = resize_inside

    """
    Get the inception features without resizing
    x: Image with values in range [0,255]
    """

    def forward(self, x):
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            if self.resize_inside:
                features = self.base(x, return_features=True).view((bs, 2048))
            else:
                # make sure it is resized already
                assert (x.shape[2] == 299) and (x.shape[3] == 299)
                # apply normalization
                x1 = x - 128
                x2 = x1 / 128
                features = self.layers.forward(
                    x2,
                ).view((bs, 2048))
            return features


####InceptionV3W End######


####Resize#####
"""
Helpers for resizing with multiple CPU cores
"""
dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX,
    },
    "OpenCV": {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    },
}


def build_resizer(mode):
    if mode == "clean":
        return make_resizer("PIL", False, "bicubic", (299, 299))
    # if using legacy tensorflow, do not manually resize outside the network
    elif mode == "legacy_tensorflow":
        return lambda x: x
    elif mode == "legacy_pytorch":
        return make_resizer("PyTorch", False, "bilinear", (299, 299))
    else:
        raise ValueError(f"Invalid mode {mode} specified")


"""
Construct a function that resizes a numpy image based on the
flags passed in. 
"""


def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:

        def func(x):
            x = Image.fromarray(x)
            x = x.resize(
                output_size, resample=dict_name_to_filter[library][filter]
            )
            x = np.asarray(x).astype(np.uint8)
            return x

    elif library == "PIL" and not quantize_after:
        s1, s2 = output_size

        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode="F")
            img = img.resize(
                output_size, resample=dict_name_to_filter[library][filter]
            )
            return np.asarray(img).reshape(s1, s2, 1)

        def func(x):
            # x = resize_single_channel(x[:, :, 0])
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x

    elif library == "PyTorch":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")

        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(
                x, size=output_size, mode=filter, align_corners=False
            )
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    elif library == "TensorFlow":
        import warnings

        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        import tensorflow as tf

        def func(x):
            x = tf.constant(x)[tf.newaxis, ...]
            x = tf.image.resize(x, output_size, method=filter)
            x = x[0, ...].numpy().clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    elif library == "OpenCV":
        import cv2

        name_to_filter = {
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }

        def func(x):
            x = cv2.resize(
                x, output_size, interpolation=name_to_filter[filter]
            )
            if quantize_after:
                x = x.astype(np.uint8)
            return x

    else:
        raise NotImplementedError("library [%s] is not include" % library)
    return func


class FolderResizer(torch.utils.data.Dataset):
    def __init__(self, files, outpath, fn_resize, output_ext=".png"):
        self.files = files
        self.outpath = outpath
        self.output_ext = output_ext
        self.fn_resize = fn_resize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        img_np = np.asarray(Image.open(path))
        img_resize_np = self.fn_resize(img_np)
        # swap the output extension
        basename = os.path.basename(path).split(".")[0] + self.output_ext
        outname = os.path.join(self.outpath, basename)
        if self.output_ext == ".npy":
            np.save(outname, img_resize_np)
        elif self.output_ext == ".png":
            img_resized_pil = Image.fromarray(img_resize_np)
            img_resized_pil.save(outname)
        else:
            raise ValueError("invalid output extension")
        return 0


####utils####


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores
    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299)):
        self.files = files
        self.transforms = torchvision.transforms.ToTensor()
        # normalize the image for the inception model
        # self.norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.size = size
        self.fn_resize = build_resizer(mode)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)

        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
    "npy",
}


####Resize &utils End#####


###Features####
"""
returns a functions that takes an image in range [0,255]
and outputs a feature embedding vector
"""


def feature_extractor(
    name="torchscript_inception",
    device=torch.device("cuda"),
    resize_inside=False,
):
    if name == "torchscript_inception":
        # model = InceptionV3W("/tmp", download=True, resize_inside=resize_inside).to(device)
        model = InceptionV3pxd(resize_inside=resize_inside).to(device)
        model.eval()

        def model_fn(x):
            return model(x)

    elif name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).to(device)
        model.eval()

        def model_fn(x):
            return model(x / 255)[0].squeeze(-1).squeeze(-1)

    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


"""
Build a feature extractor for each of the modes
"""


def build_feature_extractor(mode, device=torch.device("cuda")):
    if mode == "legacy_pytorch":
        feat_model = feature_extractor(
            name="pytorch_inception", resize_inside=False, device=device
        )
    elif mode == "legacy_tensorflow":
        feat_model = feature_extractor(
            name="torchscript_inception", resize_inside=True, device=device
        )
    elif mode == "clean":
        feat_model = feature_extractor(
            name="torchscript_inception", resize_inside=False, device=device
        )
    return feat_model


"""
Load precomputed reference statistics for commonly used datasets
"""


def get_reference_statistics(
    name, res, mode="clean", seed=0, split="test", metric="FID"
):
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats/"
    if split == "custom":
        res = "na"
    if metric == "FID":
        rel_path = (f"{name}_{mode}_{split}_{res}.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    elif metric == "KID":
        rel_path = (f"{name}_{mode}_{split}_{res}_kid.npz").lower()
        url = f"{base_url}/{rel_path}"
        mod_path = os.path.dirname(cleanfid.__file__)
        stats_folder = os.path.join(mod_path, "stats")
        fpath = check_download_url(local_folder=stats_folder, url=url)
        stats = np.load(fpath)
        return stats["feats"]


####Features_end######


"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


"""
Compute the KID score given the sets of features
"""


def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


"""
Compute the inception features for a batch of images
"""


def get_batch_features(batch, model, device):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


"""
Compute the inception features for a list of files
"""


def get_files_features(
    l_files,
    model=None,
    num_workers=12,
    batch_size=128,
    device=torch.device("cuda"),
    mode="clean",
    custom_fn_resize=None,
    description="",
):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)

    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    # collect all inception features
    l_feats = []
    for batch in tqdm(dataloader, desc=description):
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a folder of image files
"""


def get_folder_features(
    fdir,
    model=None,
    num_workers=12,
    num=None,
    shuffle=False,
    seed=0,
    batch_size=128,
    device=torch.device("cuda"),
    mode="clean",
    custom_fn_resize=None,
    description="",
):
    # get all relevant files in the dataset
    files = sorted(
        [
            file
            for ext in EXTENSIONS
            for file in glob(
                os.path.join(fdir, f"**/*.{ext}"), recursive=True
            )
        ]
    )
    print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(
        files,
        model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        custom_fn_resize=custom_fn_resize,
        description=description,
    )
    return np_feats


"""
Compute the FID score given the inception features stack
"""


def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


"""
Computes the FID score for a folder of images for a specific dataset 
and a specific resolution
"""


def fid_folder(
    fdir,
    dataset_name,
    dataset_res,
    dataset_split,
    model=None,
    mode="clean",
    num_workers=12,
    batch_size=128,
    device=torch.device("cuda"),
):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(
        dataset_name, dataset_res, mode=mode, seed=0, split=dataset_split
    )
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(
        fdir,
        model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname} : ",
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


def trunc_trick(bs, z_dim, bound=1):
    z = torch.randn(bs, z_dim)
    while z.min() < -bound or bound < z.max():
        z = z.where((-bound < z) & (z < bound), torch.randn_like(z))
    return z


def thresh(x):
    return F.threshold(x, -0.25, -1)


"""
Compute the FID stats from a generator model
"""


def get_model_features(
    G,
    model,
    mode="clean",
    z_dim=128,
    trunc=1,
    num_gen=50000,
    batch_size=40,
    device=torch.device("cuda"),
    desc="FID model: ",
):
    fn_resize = build_resizer(mode)
    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    # for idx in tqdm(range(num_iters), desc=desc):
    for idx in range(num_iters):
        with torch.no_grad():
            # labels =  torch.randint(0, 40, size=(batch_size,), dtype=torch.long, device=device)
            labels = torch.randperm(40, device=device, requires_grad=False)
            if trunc is not None:
                z_batch = trunc_trick(batch_size, z_dim, bound=trunc).to(
                    device
                )
            else:
                z_batch = torch.randn((batch_size, z_dim)).to(device)
            # generated image is in range [0,255]
            img_batch = G(z_batch, labels)
            img_batch = thresh(img_batch)
            img_batch = img_batch.mul_(0.5).add_(0.5)
            img_batch = torch.pow(256, img_batch).add_(-1).clamp_(0, 255)
            img_batch = img_batch[:, 0, 3:-3, :].unsqueeze(1)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                resized_batch = torch.zeros(batch_size, 3, 299, 299)
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    img_resize = fn_resize(img_np)
                    resized_batch[idx] = torch.tensor(
                        img_resize.transpose((2, 0, 1))
                    )
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Computes the FID score for a generator model for a specific dataset 
and a specific resolution
"""


def fid_model(
    G,
    dataset_name,
    dataset_res,
    dataset_split,
    model=None,
    z_dim=512,
    trunc=1,
    num_gen=50000,
    mode="clean",
    num_workers=0,
    batch_size=128,
    device=torch.device("cuda"),
):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(
        dataset_name, dataset_res, mode=mode, seed=0, split=dataset_split
    )
    # build resizing function based on options
    fn_resize = build_resizer(mode)

    # Generate test features
    np_feats = get_model_features(
        G,
        model,
        mode=mode,
        z_dim=z_dim,
        trunc=trunc,
        num_gen=num_gen,
        batch_size=batch_size,
        device=device,
    )

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Computes the FID score between the two given folders
"""


def compare_folders(
    fdir1,
    fdir2,
    feat_model,
    mode,
    num_workers=0,
    batch_size=8,
    device=torch.device("cuda"),
):
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(
        fdir1,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname1} : ",
    )
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(
        fdir2,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname2} : ",
    )
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


"""
Test if a custom statistic exists
"""


def test_stats_exists(name, mode):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    fname = f"{name}_{mode}_{split}_{res}.npz"
    fpath = os.path.join(stats_folder, fname)
    return os.path.exists(fpath)


"""
Remove the custom FID features from the stats folder
"""


def remove_custom_stats(name, mode="clean"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)


"""
Cache a custom dataset statistics file
"""


def make_custom_stats(
    name,
    fdir,
    num=None,
    mode="clean",
    num_workers=0,
    batch_size=64,
    device=torch.device("cuda"),
):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += f"Use remove_custom_stats function to delete it first."
        raise Exception(msg)

    feat_model = build_feature_extractor(mode, device)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(
        fdir,
        feat_model,
        num_workers=num_workers,
        num=num,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname} : ",
    )
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)


def compute_fid(
    fdir1=None,
    fdir2=None,
    gen=None,
    mode="clean",
    num_workers=12,
    batch_size=32,
    device=torch.device("cuda"),
    dataset_name="FFHQ",
    dataset_res=1024,
    dataset_split="train",
    num_gen=50000,
    z_dim=512,
    trunc=1,
):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute FID between two folders")
        score = compare_folders(
            fdir1,
            fdir2,
            feat_model,
            mode=mode,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        print(f"compute FID of a folder with {dataset_name} statistics")
        score = fid_folder(
            fdir1,
            dataset_name,
            dataset_res,
            dataset_split,
            model=feat_model,
            mode=mode,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
        )
        return score

    # compute fid for a generator
    elif gen is not None:
        print(
            f"compute FID of a model with {dataset_name}-{dataset_res} statistics"
        )
        score = fid_model(
            gen,
            dataset_name,
            dataset_res,
            dataset_split,
            model=feat_model,
            z_dim=z_dim,
            trunc=trunc,
            num_gen=num_gen,
            mode=mode,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
        )
        return score

    else:
        raise ValueError(
            f"invalid combination of directories and models entered"
        )
