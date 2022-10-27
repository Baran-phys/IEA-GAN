import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

import layers
from diff_aug import DiffAugment
# from adabelief_pytorch import AdaBelief

# relational reasoning module
import RRM


class GBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=layers.SNConv2d,
        which_bn=layers.bn,
        activation=None,
        upsample=None,
        channel_ratio=4,
    ):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        # Conv layers
        self.conv1 = self.which_conv(
            self.in_channels, self.hidden_channels, kernel_size=1, padding=0
        )
        self.conv2 = self.which_conv(
            self.hidden_channels, self.hidden_channels
        )
        self.conv3 = self.which_conv(
            self.hidden_channels, self.hidden_channels
        )
        self.conv4 = self.which_conv(
            self.hidden_channels, self.out_channels, kernel_size=1, padding=0
        )
        # Batchnorm layers
        self.bn1 = self.which_bn(self.in_channels)
        self.bn2 = self.which_bn(self.hidden_channels)
        self.bn3 = self.which_bn(self.hidden_channels)
        self.bn4 = self.which_bn(self.hidden_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, : self.out_channels]
        # Upsample both h and x at this point
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x


def G_arch(ch=64, attention="64", ksize="333333", dilation="111111"):
    arch = {}
    arch[512] = {
        "in_channels": [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
        "out_channels": [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
        "upsample": [True] * 7,
        "resolution": [8, 16, 32, 64, 128, 256, 512],
        "attention": {
            2**i: (2 ** i in [int(item) for item in attention.split("_")])
            for i in range(3, 10)
        },
    }
    arch[256] = {
        "in_channels": [ch * item for item in [16, 16, 8, 8, 4, 2]],
        "out_channels": [ch * item for item in [16, 8, 8, 4, 2, 1]],
        "upsample": [True] * 6,
        "resolution": [8, 16, 32, 64, 128, 256],
        "attention": {
            2**i: (2 ** i in [int(item) for item in attention.split("_")])
            for i in range(3, 9)
        },
    }
    arch[128] = {
        "in_channels": [ch * item for item in [16, 16, 8, 4, 2]],
        "out_channels": [ch * item for item in [16, 8, 4, 2, 1]],
        "upsample": [True] * 5,
        "resolution": [8, 16, 32, 64, 128],
        "attention": {
            2**i: (2 ** i in [int(item) for item in attention.split("_")])
            for i in range(3, 8)
        },
    }
    arch[96]  = {
        'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
        'out_channels' : [ch * item for item in [16, 8, 4, 2]],
        'upsample' : [True] * 4,
        'resolution' : [12, 24, 48, 96],
        'attention' : {
            12*2**i: (6*2**i in [int(item) for item in attention.split('_')])
            for i in range(0,4)}}

    arch[64] = {
        "in_channels": [ch * item for item in [16, 16, 8, 4]],
        "out_channels": [ch * item for item in [16, 8, 4, 2]],
        "upsample": [True] * 4,
        "resolution": [8, 16, 32, 64],
        "attention": {
            2**i: (2 ** i in [int(item) for item in attention.split("_")])
            for i in range(3, 7)
        },
    }
    arch[32] = {
        "in_channels": [ch * item for item in [4, 4, 4]],
        "out_channels": [ch * item for item in [4, 4, 4]],
        "upsample": [True] * 3,
        "resolution": [8, 16, 32],
        "attention": {
            2**i: (2 ** i in [int(item) for item in attention.split("_")])
            for i in range(3, 6)
        },
    }

    return arch


class Generator(nn.Module):
    def __init__(
        self,
        G_ch=64,
        G_depth=2,
        dim_z=128,
        bottom_width=4,
        resolution=256,
        G_kernel_size=3,
        G_attn="64",
        n_classes=40,
        H_base=1,
        num_G_SVs=1,
        num_G_SV_itrs=1,
        attn_type="sa",
        G_shared=True,
        shared_dim=128,
        rdof_dim = 4,
        hier=True,
        cross_replica=False,
        mybn=False,
        G_activation="relu",
        G_lr=5e-5,
        G_B1=0.0,
        G_B2=0.999,
        adam_eps=1e-8,
        BN_eps=1e-5,
        SN_eps=1e-12,
        G_init="ortho",
        G_mixed_precision=False,
        G_fp16=False,
        skip_init=False,
        no_optim=False,
        sched_version='default',
        RRM_prx_G=True,
        prior_embed=False,
        n_head_G=2,
        G_param="SN",
        norm_style="bn",
        device = 'cuda',
        **kwargs
    ):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Number of resblocks per stage
        self.G_depth = G_depth
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # The initial harizontal dimension
        self.H_base = H_base
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        if G_activation == "inplace_relu":
            self.activation = torch.nn.ReLU(inplace=True)
        elif G_activation == "relu":
            self.activation = torch.nn.ReLU(inplace=False)
        elif G_activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU(0.2, inplace=False)
        else:
            raise NotImplementedError(f"activation function {G_activation} not implemented")
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]
        self.RRM_prx_G = RRM_prx_G
        self.n_head_G = n_head_G
        self.prior_embed = prior_embed
        self.rdof_dim = rdof_dim
        self.device = device

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == "SN":
            self.which_conv = functools.partial(
                layers.SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_linear = functools.partial(
                layers.SNLinear,
                num_svs=num_G_SVs,
                num_itrs=num_G_SV_itrs,
                eps=self.SN_eps,
            )
        else:
            self.which_conv = functools.partial(
                nn.Conv2d, kernel_size=3, padding=1
            )
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (
            functools.partial(self.which_linear, bias=False)
            if self.G_shared
            else self.which_embedding
        )
        self.which_bn = functools.partial(
            layers.ccbn,
            which_linear=bn_linear,
            cross_replica=self.cross_replica,
            mybn=self.mybn,
            input_size=(
                self.shared_dim + self.dim_z
                if self.G_shared
                else self.n_classes
            ),
            norm_style=self.norm_style,
            eps=self.BN_eps,
        )

        # Prepare model
        #Having prior embedding
        if self.prior_embed:
            self.shared = (
                self.which_embedding(n_classes, self.shared_dim//2)
                if G_shared
                else layers.identity()
            )
            self.linear0 = self.which_linear(1, self.shared_dim//2)
            self.linear1 = self.which_linear(self.shared_dim, self.shared_dim)

        else:
            self.shared = (
                self.which_embedding(n_classes, self.shared_dim)
                if G_shared
                else layers.identity()
            )

        if self.RRM_prx_G:
            #Rdof for event generation
            self.linear_f = self.which_linear(self.shared_dim + self.rdof_dim, 128)
            #RRM on proxy embeddings
            self.RR_G = RRM.RelationalReasoning(
                num_layers=1,
                input_dim=128,
                dim_feedforward=128,
                which_linear=nn.Linear,
                num_heads=self.n_head_G,
                dropout=0.0,
                hidden_dim=128,
            )

        # First linear layer
        self.linear = self.which_linear(
            self.dim_z + self.shared_dim,
            self.arch["in_channels"][0]
            * ((self.bottom_width**2) * self.H_base),
        )

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch["out_channels"])):
            self.blocks += [
                [
                    GBlock(
                        in_channels=self.arch["in_channels"][index],
                        out_channels=self.arch["in_channels"][index]
                        if g_index == 0
                        else self.arch["out_channels"][index],
                        which_conv=self.which_conv,
                        which_bn=self.which_bn,
                        activation=self.activation,
                        upsample=(
                            functools.partial(F.interpolate, scale_factor=2)
                            if self.arch["upsample"][index]
                            and g_index == (self.G_depth - 1)
                            else None
                        ),
                    )
                ]
                for g_index in range(self.G_depth)
            ]

            # If attention on this block, attach it to the end
            if self.arch["attention"][self.arch["resolution"][index]]:
                print(
                    "Adding attention layer in G at resolution %d"
                    % self.arch["resolution"][index]
                )

                if attn_type == "sa":
                    self.blocks[-1] += [
                        layers.Attention(
                            self.arch["out_channels"][index], self.which_conv
                        )
                    ]
                elif attn_type == "cbam":
                    self.blocks[-1] += [
                        layers.CBAM_attention(
                            self.arch["out_channels"][index], self.which_conv
                        )
                    ]
                elif attn_type == "ila":
                    self.blocks[-1] += [
                        layers.ILA(self.arch["out_channels"][index])
                    ]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList(
            [nn.ModuleList(block) for block in self.blocks]
        )

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(
            layers.bn(
                self.arch["out_channels"][-1],
                cross_replica=self.cross_replica,
                mybn=self.mybn,
            ),
            self.activation,
            self.which_conv(self.arch["out_channels"][-1], 1),
        )

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print("Using fp16 adam in G...")
            import utils

            self.optim = utils.Adam16(
                params=self.parameters(),
                lr=self.lr,
                betas=(self.B1, self.B2),
                weight_decay=0,
                eps=self.adam_eps,
            )

        self.optim = optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=(self.B1, self.B2),
            weight_decay=0,
            eps=self.adam_eps,
        )
        # LR scheduling
        if sched_version=='default':
            self.lr_sched=None
        elif  sched_version=='CosAnnealLR':
            self.lr_sched =optim.lr_scheduler.CosineAnnealingLR(self.optim,
                            T_max=kwargs["num_epochs"], eta_min=self.lr/4, last_epoch=-1)
        elif  sched_version=='CosAnnealWarmRes':
            self.lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                            T_0=10, T_mult=2, eta_min=self.lr/4)
        else:
            self.lr_sched = None

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    print("Init style not recognized...")
                self.param_count += sum(
                    [p.data.nelement() for p in module.parameters()]
                )
        print(
            "Param count for G"
            "s initialized parameters: %d" % self.param_count
        )

    def forward(self, z, y):
        # If prior embedding
        if self.prior_embed:
            prs = layers.prior(y, norm=True, device=self.device)
            y = self.shared(y)
            feat = self.linear0(prs)
            y = self.linear1(torch.cat((y, feat), 1))
        else:
            y = self.shared(y)
        # If relational embedding
        if self.RRM_prx_G:
            #Rdof
            rdof = torch.randn(40, self.rdof_dim, device=self.device)
            y = self.linear_f(torch.cat([y, rdof], 1))
            y = self.RR_G(y.unsqueeze(0)).squeeze(0)
            # y = F.normalize(y, dim=1)
        # If hierarchical, concatenate zs and ys
        if self.hier:  # y and z are [bs,128] dimensional
            z = torch.cat([y, z], 1)
            y = z
        # First linear layer
        h = self.linear(z)  # ([bs,256]-->[bs,24576])
        # Reshape
        h = h.view(
            h.size(0), -1, self.bottom_width, self.bottom_width * self.H_base
        )
        # Loop over blocks
        for _, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, y)

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=layers.SNConv2d,
        wide=True,
        preactivation=True,
        activation=None,
        downsample=None,
        channel_ratio=4,
    ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels // channel_ratio
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(
            self.in_channels, self.hidden_channels, kernel_size=1, padding=0
        )
        self.conv2 = self.which_conv(
            self.hidden_channels, self.hidden_channels
        )
        self.conv3 = self.which_conv(
            self.hidden_channels, self.hidden_channels
        )
        self.conv4 = self.which_conv(
            self.hidden_channels, self.out_channels, kernel_size=1, padding=0
        )

        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(
                in_channels,
                out_channels - in_channels,
                kernel_size=1,
                padding=0,
            )

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = torch.cat([x, self.conv_sc(x)], 1)
        return x

    def forward(self, x):
        # 1x1 bottleneck conv
        h = x
        if self.preactivation:
            h = F.relu(h)
        h = self.conv1(h)
        # 3x3 convs
        h = self.conv2(self.activation(h))
        h = self.conv3(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = self.downsample(h)
        # final 1x1 conv
        h = self.conv4(h)
        return h + self.shortcut(x)


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention="64", ksize="333333", dilation="111111"):
    arch = {}
    arch[512] = {
        "in_channels": [item * ch for item in [1, 1, 2, 4, 8, 8, 16]],
        "out_channels": [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
        "downsample": [True] * 7 + [False],
        "resolution": [256, 128, 64, 32, 16, 8, 4, 4],
        "attention": {
            2**i: 2 ** i in [int(item) for item in attention.split("_")]
            for i in range(2, 10)
        },
    }
    arch[256] = {
        "in_channels": [item * ch for item in [1, 2, 4, 8, 8, 16]],
        "out_channels": [item * ch for item in [2, 4, 8, 8, 16, 16]],
        "downsample": [True] * 6 + [False],
        "resolution": [128, 64, 32, 16, 8, 4, 4],
        "attention": {
            2**i: 2 ** i in [int(item) for item in attention.split("_")]
            for i in range(2, 9)
        },
    }
    arch[128] = {
        "in_channels": [item * ch for item in [1, 2, 4, 8, 16]],
        "out_channels": [item * ch for item in [2, 4, 8, 16, 16]],
        "downsample": [True] * 5 + [False],
        "resolution": [64, 32, 16, 8, 4, 4],
        "attention": {
            2**i: 2 ** i in [int(item) for item in attention.split("_")]
            for i in range(2, 8)
        },
    }
    arch['96']  = {
        'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
        'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
        'downsample' : [True] * 4 + [False],
        'resolution' : [48, 24, 12, 6, 6],
        'attention' : {6*2**i: 6*2**i in [int(item) for item in attention.split('_')]
                              for i in range(0,4)}}

    arch[64] = {
        "in_channels": [item * ch for item in [1, 2, 4, 8]],
        "out_channels": [item * ch for item in [2, 4, 8, 16]],
        "downsample": [True] * 4 + [False],
        "resolution": [32, 16, 8, 4, 4],
        "attention": {
            2**i: 2 ** i in [int(item) for item in attention.split("_")]
            for i in range(2, 7)
        },
    }
    arch[32] = {
        "in_channels": [item * ch for item in [4, 4, 4]],
        "out_channels": [item * ch for item in [4, 4, 4]],
        "downsample": [True, True, False, False],
        "resolution": [16, 16, 16, 16],
        "attention": {
            2**i: 2 ** i in [int(item) for item in attention.split("_")]
            for i in range(2, 6)
        },
    }
    return arch


class Discriminator(nn.Module):
    def __init__(
        self,
        D_ch=64,
        D_wide=True,
        D_depth=2,
        resolution=256,
        D_kernel_size=3,
        D_attn="64",
        n_classes=40,
        attn_type="sa",
        num_D_SVs=1,
        num_D_SV_itrs=1,
        D_activation="relu",
        conditional_strategy="Proj",
        D_lr=2e-4,
        D_B1=0.0,
        D_B2=0.999,
        adam_eps=1e-8,
        SN_eps=1e-12,
        output_dim=1,
        D_init="ortho",
        D_mixed_precision=False,
        D_fp16=False,
        sched_version='default',
        skip_init=False,
        D_param="SN",
        hypersphere_dim=512,
        nonlinear_embed=False,
        normalize_embed=True,
        prior_embed=False,
        RRM_prx_D=False,
        RRM_embed=False,
        n_head_D=4,
        **kwargs
    ):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # How many resblocks per stage?
        self.D_depth = D_depth
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        if D_activation == "inplace_relu":
            self.activation = torch.nn.ReLU(inplace=True)
        elif D_activation == "relu":
            self.activation = torch.nn.ReLU(inplace=False)
        elif D_activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU(0.2, inplace=False)
        else:
            raise NotImplementedError(f"activation function {D_activation} not implemented")
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        self.RRM_prx_D = RRM_prx_D
        self.RRM_embed = RRM_embed
        self.prior_embed = prior_embed
        self.conditional_strategy = conditional_strategy
        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.RRM_prx_D = RRM_prx_D
        self.RRM_embed = RRM_embed
        self.n_head_D = n_head_D
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == "SN":
            self.which_conv = functools.partial(
                layers.SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_linear = functools.partial(
                layers.SNLinear,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )
            self.which_embedding = functools.partial(
                layers.SNEmbedding,
                num_svs=num_D_SVs,
                num_itrs=num_D_SV_itrs,
                eps=self.SN_eps,
            )

        # Prepare model
        # Stem convolution
        self.input_conv = self.which_conv(1, self.arch["in_channels"][0])
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch["out_channels"])):
            self.blocks += [
                [
                    DBlock(
                        in_channels=self.arch["in_channels"][index]
                        if d_index == 0
                        else self.arch["out_channels"][index],
                        out_channels=self.arch["out_channels"][index],
                        which_conv=self.which_conv,
                        wide=self.D_wide,
                        activation=self.activation,
                        preactivation=index > 0 or d_index > 0,
                        downsample=(
                            nn.AvgPool2d(2)
                            if self.arch["downsample"][index] and d_index == 0
                            else None
                        ),
                    )
                    for d_index in range(self.D_depth)
                ]
            ]
            # If attention on this block, attach it to the end
            if self.arch["attention"][self.arch["resolution"][index]]:
                print(
                    "Adding attention layer in D at resolution %d"
                    % self.arch["resolution"][index]
                )
                if attn_type == "sa":
                    self.blocks[-1] += [
                        layers.Attention(
                            self.arch["out_channels"][index], self.which_conv
                        )
                    ]
                elif attn_type == "cbam":
                    self.blocks[-1] += [
                        layers.CBAM_attention(
                            self.arch["out_channels"][index], self.which_conv
                        )
                    ]
                elif attn_type == "ila":
                    self.blocks[-1] += [
                        layers.ILA(self.arch["out_channels"][index])
                    ]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList(
            [nn.ModuleList(block) for block in self.blocks]
        )
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear0 = self.which_linear(
            self.arch["out_channels"][-1], output_dim
        )

        if self.RRM_embed:
            self.RR_D = RRM.RelationalReasoning(
                num_layers=1,
                input_dim=self.arch["out_channels"][-1],
                dim_feedforward=512,
                num_heads=self.n_head_D,
                dropout=0.0,
                hidden_dim=512,
                which_linear=self.which_linear,
            )
            self.norm = nn.LayerNorm(hypersphere_dim)

        if self.conditional_strategy == "Proj":
            # Embedding for projection discrimination
            self.embed = self.which_embedding(
                self.n_classes, self.arch["out_channels"][-1]
            )

        if self.conditional_strategy == "Contra":
            self.linear1 = self.which_linear(
                self.arch["out_channels"][-1], hypersphere_dim
            )  # D_ch * 16
            if self.RRM_prx_D:
                # self.multihead_attn = nn.MultiheadAttention(512, self.n_head) nn.Linear
                # which_linear=self.which_linear
                self.RR_Dproxy = RRM.RelationalReasoning(
                    num_layers=1,
                    input_dim=hypersphere_dim,
                    dim_feedforward=hypersphere_dim,
                    which_linear=self.which_linear,
                    num_heads=self.n_head_D,
                    dropout=0.0,
                    hidden_dim=hypersphere_dim,
                )

            if self.nonlinear_embed:
                self.linear2 = self.which_linear(
                    hypersphere_dim, hypersphere_dim
                )
            if self.prior_embed:
                self.embed = self.which_embedding(
                    self.n_classes, hypersphere_dim // 2
                )  # division by two for new features
                self.linear3 = self.which_linear(1, hypersphere_dim // 2)
                self.linear4 = self.which_linear(
                    hypersphere_dim, hypersphere_dim
                )  # hypersphere_dim//2
            else:
                self.embed = self.which_embedding(
                    self.n_classes, hypersphere_dim
                )

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print("Using fp16 adam in D...")
            import utils

            self.optim = utils.Adam16(
                params=self.parameters(),
                lr=self.lr,
                betas=(self.B1, self.B2),
                weight_decay=0,
                eps=self.adam_eps,
            )

        self.optim = optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=(self.B1, self.B2),
            weight_decay=0,
            eps=self.adam_eps,
        )
        # LR scheduling
        if sched_version=='default':
            self.lr_sched=None
        elif  sched_version=='CosAnnealLR':
            self.lr_sched =optim.lr_scheduler.CosineAnnealingLR(self.optim,
                            T_max=kwargs["num_epochs"], eta_min=self.lr/4, last_epoch=-1)
        elif  sched_version=='CosAnnealWarmRes':
            self.lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                            T_0=10, T_mult=2, eta_min=self.lr/4)
        else:
            self.lr_sched = None

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
            ):
                if self.init == "ortho":
                    init.orthogonal_(module.weight)
                elif self.init == "N02":
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ["glorot", "xavier"]:
                    init.xavier_uniform_(module.weight)
                else:
                    print("Init style not recognized...")
                self.param_count += sum(
                    [p.data.nelement() for p in module.parameters()]
                )
        print(
            "Param count for D"
            "s initialized parameters: %d" % self.param_count
        )

    def forward(self, x, y=None):
        # pylint: disable=no-else-return
        # Run input conv
        h = self.input_conv(x)
        # Loop over blocks
        for _, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        # h size before sum pooling [bs,512,4,12]
        h = torch.sum(self.activation(h), [2, 3])  # [bs,512]

        if self.conditional_strategy == "Contra":
            out = torch.squeeze(self.linear0(h))
            cls_proxy = self.embed(y)
            if self.RRM_embed:
                h = self.RR_D(h.unsqueeze(0)).squeeze(0)
                # out = torch.squeeze(self.linear0(h))
                cls_embed = self.linear1(h)
                cls_embed = self.norm(cls_embed)
            else:
                # out = torch.squeeze(self.linear0(h))
                cls_embed = self.linear1(h)
            if self.prior_embed:
                prs = layers.prior(y, device='cuda', norm=True)
                feat = self.linear3(prs)
                cls_proxy = self.linear4(torch.cat((cls_proxy, feat), 1))
            if self.RRM_prx_D:
                cls_proxy = self.RR_Dproxy(cls_proxy.unsqueeze(0)).squeeze(0)
            if self.nonlinear_embed:
                cls_embed = self.linear2(self.activation(cls_embed))
            if self.normalize_embed:
                cls_proxy = F.normalize(cls_proxy, dim=1)
                cls_embed = F.normalize(cls_embed, dim=1)

            return cls_proxy, cls_embed, out

        elif self.conditional_strategy == "Proj":
            # Get initial class-unconditional output
            out = self.linear0(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
            return out


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    # pylint: disable=no-else-return
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(
        self,
        z,
        gy,
        x=None,
        dy=None,
        x_aug=None,
        contra=True,
        train_G=False,
        return_G_z=False,
        split_D=False,
        diff_aug=True,
        pixel_reg=False,
    ):

        policy = "color,translation,cutout"
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            G_z = self.G(z, gy)
            # Having Differentiable Augmentation
            if diff_aug:
                G_z = DiffAugment(G_z, policy=policy)
            # Having Pixel_regularization (not yet differentiable!)
            if pixel_reg:
                G_reg = F.threshold(G_z, -0.25, -1)

        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        if split_D:
            if contra:
                cls_proxies_fake, cls_embed_fake, D_fake = self.D(G_z, gy)
                # if x is not None:
                if train_G:
                    if return_G_z:
                        return (
                            cls_proxies_fake,
                            cls_embed_fake,
                            D_fake,
                            G_z,
                            G_reg,
                        )
                    else:
                        return cls_proxies_fake, cls_embed_fake, D_fake

                else:
                    cls_proxies_real, cls_embed_real, D_real = self.D(x, dy)
                    return (
                        cls_proxies_fake,
                        cls_embed_fake,
                        D_fake,
                        cls_proxies_real,
                        cls_embed_real,
                        D_real,
                    )
            else:
                D_fake = self.D(G_z, gy)
                if x is not None:
                    D_real = self.D(x, dy)
                    return D_fake, D_real
                else:
                    if return_G_z:
                        return D_fake, G_z, G_reg
                    else:
                        return D_fake
        # If real data is provided, concatenate it with the Generator's output
        # along the batch dimension for improved efficiency.
        else:
            if contra:
                if x_aug is not None:
                    D_input = (
                        torch.cat([G_z, x, x_aug], 0)
                        if x is not None
                        else G_z
                    )
                    D_class = (
                        torch.cat([gy, dy, dy], 0) if dy is not None else gy
                    )

                else:
                    D_input = torch.cat([G_z, x], 0) if x is not None else G_z
                    D_class = torch.cat([gy, dy], 0) if dy is not None else gy

                # Get Discriminator output
                cls_proxies, cls_embed, D_out = self.D(D_input, D_class)

                if x is not None:
                    if x_aug is not None:
                        D_fake, D_real, D_real_aug = torch.split(
                            D_out, [G_z.shape[0], x.shape[0], x_aug.shape[0]]
                        )
                        (
                            cls_embed_fake,
                            cls_embed_real,
                            cls_embed_real_aug,
                        ) = torch.split(
                            cls_embed,
                            [G_z.shape[0], x.shape[0], x_aug.shape[0]],
                        )
                        cls_proxies_fake, cls_proxies_real, _ = torch.split(
                            cls_proxies,
                            [G_z.shape[0], x.shape[0], x_aug.shape[0]],
                        )
                        return (
                            cls_proxies_fake,
                            cls_embed_fake,
                            D_fake,
                            cls_proxies_real,
                            cls_embed_real,
                            D_real,
                            cls_embed_real_aug,
                            D_real_aug,
                        )
                    else:
                        D_fake, D_real = torch.split(
                            D_out, [G_z.shape[0], x.shape[0]]
                        )
                        cls_embed_fake, cls_embed_real = torch.split(
                            cls_embed, [G_z.shape[0], x.shape[0]]
                        )
                        cls_proxies_fake, cls_proxies_real = torch.split(
                            cls_proxies, [G_z.shape[0], x.shape[0]]
                        )
                        return (
                            cls_proxies_fake,
                            cls_embed_fake,
                            D_fake,
                            cls_proxies_real,
                            cls_embed_real,
                            D_real,
                        )
                else:
                    if return_G_z:
                        return cls_proxies, cls_embed, D_out, G_z, G_reg
                    else:
                        return cls_proxies, cls_embed, D_out
            else:
                if x_aug is not None:
                    D_input = (
                        torch.cat([G_z, x, x_aug], 0)
                        if x is not None
                        else G_z
                    )
                    D_class = (
                        torch.cat([gy, dy, dy], 0) if dy is not None else gy
                    )
                else:
                    D_input = torch.cat([G_z, x], 0) if x is not None else G_z
                    D_class = torch.cat([gy, dy], 0) if dy is not None else gy
                # Get Discriminator output
                D_out = self.D(D_input, D_class)
                if x is not None:
                    if x_aug is not None:
                        D_fake, D_real, D_real_aug = torch.split(
                            D_out, [G_z.shape[0], x.shape[0], x_aug.shape[0]]
                        )
                        return (D_fake, D_real, D_real_aug)
                    else:
                        return torch.split(
                            D_out, [G_z.shape[0], x.shape[0]]
                        )  # D_fake, D_real
                else:
                    if return_G_z:
                        return D_out, G_z, G_reg
                    else:
                        return D_out


class Model(Generator):
    def __init__(self, config: dict):
        assert isinstance(config, dict), "Expected configuration dictionary"
        super().__init__(**config)


def generate(model):
    device = next(model.parameters()).device
    with torch.no_grad():
        latents = torch.randn(40, 128, device=device)
        labels = torch.tensor(
            [c for c in range(40)],
            dtype=torch.long,
            device=device
        )
        imgs = model(latents, labels).detach().cpu()
        # Cut the noise below 7 ADU
        imgs = F.threshold(imgs, -0.26, -1)
        # center range [-1, 1] to [0, 1]
        imgs = imgs.mul_(0.5).add_(0.5)
        # renormalize and convert to uint8
        imgs = torch.pow(256, imgs).add_(-1).clamp_(0, 255)#.to(torch.uint8)
        # flatten channel dimension and crop 256 to 250
        imgs = imgs[:, 0, 3:-3, :]
        return imgs
