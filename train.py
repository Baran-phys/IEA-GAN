"""
Main script to train the IEAGAN model
"""
import argparse
import datetime
import json

import torch
from tqdm import tqdm

# Import my stuff
import layers
import model
import train_fns
import utils
import utils.configuration as cf
from utils.logging import MetricsLogger, Logger
from utils.dataloader import load_dataset
from utils.plot import plot_sim_heatmap


def run(config: dict):
    """
    The main training file. Config is a dictionary specifying the configuration
    of this training run.

    Args:
        config (dict): settings
    """
    # By default, skip init if resuming training.
    if config["resume"]:
        print("Skipping initialization for training resumption...")
        config["skip_init"] = True

    device = config["device"]

    # Seed RNG
    utils.seed_rng(config["seed"])

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config["ema"]:
        print("Preparing EMA for G with decay of {}".format(config["ema_decay"]))
        G_ema = model.Generator(**{**config, "skip_init": True, "no_optim": True}).to(device)
        ema = utils.apply_ema(G, G_ema, config["ema_decay"], config["ema_start"])
    else:
        G_ema, ema = None, None

    # FP16?
    if config["G_fp16"]:
        print("Casting G to float16...")
        G = G.half()
        if config["ema"]:
            G_ema = G_ema.half()
    if config["D_fp16"]:
        print("Casting D to fp16...")
        D = D.half()

    GD = model.G_D(G, D)
    # print(G)
    # print(D)
    print("Number of params in G: {} D: {}".format(*[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {
        "itr": 0,
        "epoch": 0,
        "save_num": 0,
        "save_best_num": 0,
        "best_FID": 999999,
    }

    # If loading from a pre-trained model, load weights
    if config["resume"]:
        print("Loading weights...")
        utils.load_weights(
            G,
            D,
            state_dict,
            config,
            weight_name=config["load_weights"] if config["load_weights"] else None,
            G_ema=G_ema if config["ema"] else None,
            load_optim=config["load_optim"]
        )

    if G.lr_sched is not None:
        G.lr_sched.step(state_dict["epoch"])
    if D.lr_sched is not None:
        D.lr_sched.step(state_dict["epoch"])

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    test_log = MetricsLogger(config)
    print(f"Inception Metrics will be saved to {test_log.metriclogpath.absolute()}")
    train_log = Logger(config)
    print(f"Training Metrics will be saved to {train_log.logroot.absolute()}")

    # Write metadata
    utils.write_metadata(config, state_dict)

    # Prepare dataloader
    loader = load_dataset(
        config["dataroot"],
        config["num_workers"],
        config["shuffle"]
    )

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config["G_batch_size"], config["batch_size"])

    z_, y_ = utils.prepare_z_y(
        G_batch_size,
        G.dim_z,
        config["n_classes"],
        device=device,
        fp16=config["G_fp16"],
        z_dist=config["z_dist"],
        threshold=config["truncated_threshold"],
        y_dist="permuted",
        ngd=False,
        fixed=False,
    )
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(
        G_batch_size,
        G.dim_z,
        config["n_classes"],
        device=device,
        fp16=config["G_fp16"],
        z_dist=config["z_dist"],
        threshold=config["truncated_threshold"],
        y_dist="permuted",
        ngd=False,
        fixed=True,
    )
    fixed_z.sample_()
    fixed_y.sample_()
    # fixed_y = torch.randperm(40, device=device, requires_grad=False)
    print(f"The fixed_y is: {fixed_y}")
    # Loaders are loaded, prepare the training function
    if config["debug"]:
        # debugging, use the dummy train fn
        train = train_fns.dummy_training_function()
    else:
        train = train_fns.GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config, device)

    print("Beginning training at epoch %d..." % state_dict["epoch"])
    start_time = datetime.datetime.now()
    total_iters = config["num_epochs"] * len(loader)

    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict["epoch"], config["num_epochs"]):
        pbar = tqdm(loader)
        for _, (x, y) in enumerate(pbar):
            # Increment the iteration counter
            state_dict["itr"] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config["ema"]:
                G_ema.train()
            if config["D_fp16"]:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)
            train_log.log(state_dict["itr"], **metrics)

            # Every sv_log_interval, log singular values
            if (config["sv_log_interval"] > 0) and (not (state_dict["itr"] % config["sv_log_interval"])):
                train_log.log(state_dict["itr"], **{**utils.get_singular_values(G, "G"), **utils.get_singular_values(D, "D")})

            if not (state_dict["itr"] % config["log_interval"]):
                curr_time = datetime.datetime.now()
                elapsed = curr_time - start_time
                log = "[{}] [{}] [{} / {}] Ep {}, ".format(
                    curr_time.strftime("%H:%M:%S"), elapsed, state_dict["itr"], total_iters, epoch
                ) + ", ".join(["%s : %+4.3f" % (key, metrics[key]) for key in metrics])
                print(log)

            # Save weights and copies as configured at specified interval
            if not (state_dict["itr"] % config["save_every"]):
                if config["G_eval_mode"]:
                    print("Switchin G to eval mode...")
                    G.eval()
                    if config["ema"]:
                        G_ema.eval()
                utils.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config)
                with torch.no_grad():
                    classes = torch.tensor([c for c in range(config["n_classes"])], dtype=torch.long).to(device)
                    shared = G.shared
                    sha = shared(classes)
                    if config["prior_embed"]:
                        prs = layers.prior(classes, norm=False)
                        feat = G.linear0(prs)
                        sha = G.linear1(torch.cat((sha, feat), 1))
                    if config["RRM_prx_G"]:
                        sha = G.RR_G(sha.unsqueeze(0)).squeeze(0)
                        #sha = F.normalize(sha, dim=1)

                    embedding_layer = D.embed
                    cls_proxy = embedding_layer(classes)
                    cos_sim = torch.nn.CosineSimilarity(dim=-1)
                    sim_p = cos_sim(cls_proxy.unsqueeze(1), cls_proxy.unsqueeze(0))
                    sim_g = cos_sim(sha.unsqueeze(1), sha.unsqueeze(0))
                    plot_sim_heatmap(
                        sim_p.detach().cpu().numpy(),
                        classes.detach().cpu().numpy(),
                        classes.detach().cpu().numpy(),
                        mode="prx",
                        configuration=config,
                        state_dict=state_dict
                    )
                    plot_sim_heatmap(
                        sim_g.detach().cpu().numpy(),
                        classes.detach().cpu().numpy(),
                        classes.detach().cpu().numpy(),
                        mode="g_emb",
                        configuration=config,
                        state_dict=state_dict

                    )

            # Test every specified interval
            if not (state_dict["itr"] % config["test_every"]):
                if config["G_eval_mode"]:
                    print("Switchin G to eval mode...")
                    G.eval()
                train_fns.test(G, D, G_ema, state_dict, config, test_log)

            # if config['stop_after'] > 0 and int(time.perf_counter() - start_time) > config['stop_after']:
            # print("Time limit reached! Stopping training!")
            # return

        # Increment epoch counter at end of epoch
        state_dict["epoch"] += 1
        if G.lr_sched is not None:
            G.lr_sched.step()
        if D.lr_sched is not None:
            D.lr_sched.step()


def main(configuration: dict):
    """
    main function
    executes different jobs depending on the 'mode' setting
    """
    # Initialize run directories
    # Also dumps the current configuration
    cf.initialize_directories(configuration)

    run(configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BigGAN Deep",
        "Add common arguments for model training.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config", "--c", help="Path to JSON config file.", default="./config.json"
    )
    parser.add_argument("--outputroot", required=True, help="path to output")
    parser.add_argument("--run-name", "-n", type=str, help="name of experiment")

    parser.add_argument("--device", type=str, help="select device")

    ### Dataset/Dataloader stuff ###
    parser.add_argument("--dataroot", required=True, help="path to dataset")
    parser.add_argument("--augment", type=int, help="Augment with random crops and flips")
    parser.add_argument("--num_workers", type=int, help="number of workers for data loading")
    parser.add_argument(
        "--no_pin_memory",
        action="store_false",
        dest="pin_memory",
        help="Pin data into memory through dataloader?",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data",
    )

    ### Model stuff ###
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model module",
    )
    parser.add_argument(
        "--G_param",
        type=str,
        help="Parameterization style to use for G, spectral norm (SN) or SVD (SVD) or None",
    )
    parser.add_argument(
        "--D_param",
        type=str,
        help="Parameterization style to use for D, spectral norm (SN) or SVD (SVD) or None",
    )
    parser.add_argument("--G_ch", type=int, help="Channel multiplier for G")
    parser.add_argument("--D_ch", type=int, help="Channel multiplier for D")
    parser.add_argument(
        "--G_depth",
        type=int,
        help="Number of resblocks per stage in G",
    )
    parser.add_argument(
        "--D_depth",
        type=int,
        help="Number of resblocks per stage in D",
    )
    parser.add_argument(
        "--D_wide",
        help="Use the BigGAN or SN-GAN channel pattern for D?",
    )
    parser.add_argument(
        "--G_shared",
        action="store_true",
        help="Use shared embeddings in G?",
    )
    parser.add_argument(
        "--shared_dim",
        type=int,
        help="Gs shared embedding dimensionality; if 0, will be equal to dim_z.",
    )
    parser.add_argument("--dim_z", type=int, help="Noise dimensionality")
    parser.add_argument("--z_var", type=float, help="Noise variance")
    parser.add_argument(
        "--hier",
        action="store_true",
        help="Use hierarchical z in G",
    )
    parser.add_argument(
        "--cross_replica",
        action="store_true",
        help="Cross_replica batchnorm in G",
    )
    parser.add_argument(
        "--mybn",
        action="store_true",
        help="Use my batchnorm (which supports standing stats?)",
    )
    parser.add_argument("--G_nl", type=str, help="Activation function for G")
    parser.add_argument("--D_nl", type=str, help="Activation function for D")
    parser.add_argument(
        "--G_attn",
        type=str,
        help="What resolutions to use attention on for G (underscore separated)",
    )
    parser.add_argument(
        "--D_attn",
        type=str,
        help="What resolutions to use attention on for D (underscore separated)",
    )
    parser.add_argument(
        "--norm_style",
        type=str,
        help="Normalizer style for G, one of bn [batchnorm], in [instancenorm], "
        "ln [layernorm], gn [groupnorm]",
    )
    parser.add_argument("--bottom_width", type=int, help="Bottom width for G")
    parser.add_argument(
        "--add_blur",
        action="store_true",
        help="Add blur to Generator?",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to Generator",
    )
    parser.add_argument(
        "--add_style",
        action="store_true",
        help="Add style like StyleGAN",
    )
    parser.add_argument("--latent_op", help="use latent optimization as with NGD")
    parser.add_argument(
        "--latent_op_weight",
        type=int,
        help="In case of latent optimization, this is the weight of regularization",
    )
    parser.add_argument("--conditional_strategy", type=str, help="use Contra or Proj")
    parser.add_argument(
        "--hypersphere_dim",
        type=int,
        help="In case of Contra, the n-sphere dimension",
    )
    parser.add_argument(
        "--pos_collected_numerator",
        help="In case of having unique events this is false",
    )
    parser.add_argument("--nonlinear_embed", help="use non linear embedding as in SimCLR")
    parser.add_argument(
        "--normalize_embed",
        help="l2 (hypersphere mapping) normalization of the discriminator embeddings",
    )
    parser.add_argument("--inv_stereographic", help="use inverse stereographic projection")
    parser.add_argument(
        "--contra_lambda", type=int, help="lambda coefficient of the ContraGAN loss"
    )
    parser.add_argument("--IEA_loss", action="store_true", help="use IEA loss?")
    parser.add_argument("--IEA_lambda", type=int, help="lambda coefficient of the IEA Loss")
    parser.add_argument(
        "--Uniformity_loss",
        action="store_true",
        help="use Uniformity loss?",
    )
    parser.add_argument("--unif_lambda", type=int, help="lambda coefficient of the Uniformity Loss")
    parser.add_argument(
        "--diff_aug",
        action="store_true",
        help="use Differentiable Augmentation?",
    )
    parser.add_argument(
        "--Con_reg",
        action="store_true",
        help="use Consistancy regularization?",
    )
    parser.add_argument(
        "--Con_reg_lambda",
        type=int,
        help="lambda coefficient of the Consistancy regularization",
    )
    parser.add_argument(
        "--pixel_reg",
        action="store_true",
        help="use Pixel regularization?",
    )
    parser.add_argument(
        "--pixel_reg_lambda",
        type=int,
        help="lambda coefficient of the Pixel regularization",
    )
    parser.add_argument(
        "--RRM_prx_G",
        action="store_true",
        help="use RRM over the Generator's proxy embedding",
    )
    parser.add_argument(
        "--normalized_proxy_G",
        action="store_true",
        help="l2 nomalization of the Generator's proxy embedding",
    )
    parser.add_argument(
        "--RRM_prx_D",
        action="store_true",
        help="use RRM over the Discriminator's proxy embedding",
    )
    parser.add_argument(
        "--RRM_embed",
        action="store_true",
        help="use RRM over Discriminator's image embedding",
    )
    parser.add_argument(
        "--n_head_G", type=int, help="Number of heads for the Generator's RRM_prx_G"
    )
    parser.add_argument(
        "--rdof_dim", type=int, help="The random degrees of freedom"
    )
    parser.add_argument(
        "--n_head_D", type=int, help="Number of heads for the Discriminator's RRM_embed"
    )
    parser.add_argument(
        "--prior_embed", action="store_true", help="use prior embedding as in PE-GAN"
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        help="Attention style one of sa [non local]," "cbam [cbam] or ila [linear]",
    )
    parser.add_argument(
        "--sched_version",
        type=str,
        help="Optim version default[keep the lr as initial], " "CosAnnealLR, CosAnnealWarmRes",
    )
    parser.add_argument(
        "--z_dist",
        type=str,
        help="z sample from distribution, one of normal [normal distribution], "
        "censored_normal [Censored Normal]  "
        "bernoulli [Bernoulli]  ",
    )
    parser.add_argument(
        "--arch",
        type=str,
        help="if None, use image_size to select arch",
    )

    ### Model init stuff ###
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use; affects both initialization and dataloading.",
    )
    parser.add_argument("--G_init", type=str, help="Init style to use for G")
    parser.add_argument("--D_init", type=str, help="Init style to use for D")
    parser.add_argument(
        "--skip_init",
        action="store_true",
        help="Skip initialization, ideal for testing when ortho init was used ",
    )

    ### Optimizer stuff ###
    parser.add_argument(
        "--G_lr",
        type=float,
        help="Learning rate to use for Generator",
    )
    parser.add_argument(
        "--D_lr",
        type=float,
        help="Learning rate to use for Discriminator",
    )
    parser.add_argument("--G_B1", type=float, help="Beta1 to use for Generator")
    parser.add_argument(
        "--D_B1",
        type=float,
        help="Beta1 to use for Discriminator",
    )
    parser.add_argument(
        "--G_B2",
        type=float,
        help="Beta2 to use for Generator",
    )
    parser.add_argument(
        "--D_B2",
        type=float,
        help="Beta2 to use for Discriminator",
    )
    parser.add_argument("--truncated_threshold", type=float)
    parser.add_argument("--clip_norm", type=float)
    parser.add_argument("--amsgrad", action="store_true")

    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Default overall batchsize",
    )
    parser.add_argument(
        "--G_batch_size",
        type=int,
        help="Batch size to use for G; if 0, same as D",
    )
    parser.add_argument(
        "--num_G_accumulations",
        type=int,
        help="Number of passes to accumulate G's gradients over.",
    )
    parser.add_argument(
        "--num_D_steps",
        type=int,
        help="Number of D steps per G step",
    )
    parser.add_argument(
        "--num_D_accumulations",
        type=int,
        help="Number of passes to accumulate D's gradients over.",
    )
    parser.add_argument(
        "--split_D",
        action="store_true",
        help="Run D twice rather than concatenating inputs?",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train with multiple GPUs",
    )
    parser.add_argument(
        "--G_fp16",
        action="store_true",
        help="Train with half-precision in G?",
    )
    parser.add_argument(
        "--D_fp16",
        action="store_true",
        help="Train with half-precision in D?",
    )
    parser.add_argument(
        "--D_mixed_precision",
        action="store_true",
        help="Train with half-precision activations but fp32 params in D? ",
    )
    parser.add_argument(
        "--G_mixed_precision",
        action="store_true",
        help="Train with half-precision activations but fp32 params in G? ",
    )
    parser.add_argument(
        "--accumulate_stats",
        action="store_true",
        help='Accumulate "standing" batchnorm stats?',
    )
    parser.add_argument(
        "--num_standing_accumulations",
        type=int,
        help="Number of forward passes to use in accumulating standing stats? ",
    )

    ### Bookkeping stuff ###
    parser.add_argument(
        "--G_eval_mode",
        action="store_true",
        help="Run G in eval mode (running/standing stats?) at sample/test time? ",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        help="Save every X iterations",
    )
    parser.add_argument(
        "--num_save_copies",
        type=int,
        help="How many copies to save",
    )
    parser.add_argument(
        "--num_best_copies",
        type=int,
        help="How many previous best checkpoints to save",
    )
    parser.add_argument(
        "--which_best",
        type=str,
        help="Which metric to use to determine when to save new best checkpoints",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        help="Test every X iterations",
    )
    parser.add_argument(
        "--num_incep_images",
        type=int,
        help="Number of samples to compute inception metrics with",
    )
    parser.add_argument(
        "--hashname",
        action="store_true",
        help="Use a hash of the experiment name instead of the full config ",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Optionally override the automatic experiment naming with this arg. ",
    )
    parser.add_argument(
        "--config_from_name",
        action="store_true",
        help="Use a hash of the experiment name instead of the full config ",
    )

    ### EMA Stuff ###
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Keep an ema of G's weights?",
    )
    parser.add_argument("--ema_decay", type=float, help="EMA decay rate")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use the EMA parameters of G for evaluation?",
    )
    parser.add_argument(
        "--ema_start",
        type=int,
        help="When to start updating the EMA weights",
    )

    ### Numerical precision and SV stuff ###
    parser.add_argument(
        "--adam_eps",
        type=float,
        help="epsilon value to use for Adam",
    )
    parser.add_argument(
        "--BN_eps",
        type=float,
        help="epsilon value to use for BatchNorm",
    )
    parser.add_argument(
        "--SN_eps",
        type=float,
        help="epsilon value to use for Spectral Norm",
    )
    parser.add_argument(
        "--num_G_SVs",
        type=int,
        help="Number of SVs to track in G",
    )
    parser.add_argument(
        "--num_D_SVs",
        type=int,
        help="Number of SVs to track in D",
    )
    parser.add_argument("--num_G_SV_itrs", type=int, help="Number of SV itrs in G")
    parser.add_argument("--num_D_SV_itrs", type=int, help="Number of SV itrs in D")

    ### Ortho reg stuff ###
    parser.add_argument(
        "--G_ortho",
        type=float,
        help="Modified ortho reg coefficient in G",
    )
    parser.add_argument(
        "--D_ortho",
        type=float,
        help="Modified ortho reg coefficient in D",
    )
    parser.add_argument(
        "--toggle_grads",
        action="store_true",
        help="Toggle D and G" 's "requires_grad" settings when not training them? ',
    )

    ### Which train function ###
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Choose dummy function over real training function."
    )

    ### Resume training stuff
    parser.add_argument(
        "--load-weights",
        type=str,
        help="Suffix for which weights to load (e.g. best0, copy0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training?",
    )

    ### Log stuff ###
    parser.add_argument(
        "--logstyle",
        type=str,
        help="What style to use when logging training metrics?"
        "One of: #.#f/ #.#e (float/exp, text),"
        "pickle (python pickle),"
        "npz (numpy zip),"
        "mat (MATLAB .mat file)",
    )
    parser.add_argument(
        "--log_G_spectra",
        action="store_true",
        help="Log the top 3 singular values in each SN layer in G?",
    )
    parser.add_argument(
        "--log_D_spectra",
        action="store_true",
        help="Log the top 3 singular values in each SN layer in D?",
    )
    parser.add_argument(
        "--sv_log_interval",
        type=int,
        help="Iteration interval for logging singular values",
    )

    # parse arguments
    args = parser.parse_args()

    # load base config file
    with open(args.config, "r", encoding="utf-8") as config_fp:
        _config = json.load(config_fp)

    # Overwrite config with command line arguments
    _config.update(vars(args))

    main(_config)
