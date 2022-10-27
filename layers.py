""" Layers
    This file contains various layers for the BigGAN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P

# from gsa_pytorch import GSA
from torch.nn.modules.utils import _pair
import pandas as pd


#7 is qed bkg and 8 is mean occ. They have high correlations
def prior(
    y, device="cuda", norm=True
): 
    df = pd.read_csv("features.csv")
    dp = pd.DataFrame(df.iloc[:, 8])
    d = dp.T.to_dict("list")
    l = []
    for k in y.cpu().numpy():
        l.append(d[k])
    if norm:
        out = F.normalize(torch.FloatTensor(l), dim=0).to(device)
    else:
        out = torch.FloatTensor(l).to(device)
    return out


class LocallyConnected2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        output_size,
        kernel_size,
        stride,
        bias=False,
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size**2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, tensor):
        return tensor


# Spectral normalization base class
class SN(object):
    # pylint: disable=no-member
    def __init__(
        self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12
    ):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer("u%d" % i, torch.randn(1, num_outputs))
            self.register_buffer("sv%d" % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, "u%d" % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, "sv%d" % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps
            )
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(
            x,
            self.W_(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        num_svs=1,
        num_itrs=1,
        eps=1e-12,
    ):
        nn.Embedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
        )
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

    def forward(self, x):
        return F.embedding(x, self.W_())


class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name="attention"):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False
        )
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False
        )
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False
        )
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False
        )
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(
                -1, self.ch // 2, x.shape[2], x.shape[3]
            )
        )
        return self.gamma * o + x


class AttentionApproximation(nn.Module):
    def __init__(
        self,
        ch,
        n_hashes,
        q_cluster_size,
        k_cluster_size,
        q_attn_size=None,
        k_attn_size=None,
        max_iters=10,
        r=1,
        clustering_algo="lsh",
        progress=False,
        which_conv=SNConv2d,
        name="attention",
    ):
        """
        SmyrfAttention for BigGAN.
        """
        super(AttentionApproximation, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv

        # queries
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False
        )

        # keys
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False
        )
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False
        )
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False
        )
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.0), requires_grad=True)
        raise NotImplementedError("SmyrfAttention is not implemented please fix.")
        # self.smyrf = SmyrfAttention(
        #     n_hashes=n_hashes,
        #     q_cluster_size=q_cluster_size,
        #     k_cluster_size=k_cluster_size,
        #     q_attn_size=q_attn_size,
        #     k_attn_size=k_attn_size,
        #     max_iters=max_iters,
        #     clustering_algo=clustering_algo,
        #     r=r,
        # )
        # self.progress = progress

    def forward(self, x, y=None, return_attn_map=False):
        # Apply convs
        queries = self.theta(x)
        keys = F.max_pool2d(self.phi(x), [2, 2])
        values = F.max_pool2d(self.g(x), [2, 2])

        # Perform reshapes
        queries = queries.view(
            -1, self.ch // 8, x.shape[2] * x.shape[3]
        ).transpose(-2, -1)
        keys = keys.view(
            -1, self.ch // 8, x.shape[2] * x.shape[3] // 4
        ).transpose(-2, -1)
        values = values.view(
            -1, self.ch // 2, x.shape[2] * x.shape[3] // 4
        ).transpose(-2, -1)

        if not return_attn_map:
            out = self.smyrf(
                queries, keys, values, progress=self.progress
            ).transpose(-2, -1)
        else:
            out, attn_map = self.smyrf(
                queries,
                keys,
                values,
                progress=self.progress,
                return_attn_map=True,
            )
            out = out.transpose(-2, -1)

        o = self.o(out.reshape(x.shape[0], -1, x.shape[2], x.shape[3]))

        if not return_attn_map:
            return self.gamma * o + x
        return self.gamma * o + x, attn_map


class CBAM_attention(nn.Module):

    def __init__(self, channels, which_conv=SNConv2d,reduction=8, attention_kernel_size=3):
        super(CBAM_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = which_conv(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = which_conv(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = which_conv(2, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x, y=None,style=None):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        # b, c, h, w = x.size()
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x

# Image_Linear_Attention
class ILA(nn.Module):
    def __init__(
        self,
        chan,
        chan_out=None,
        kernel_size=1,
        padding=0,
        stride=1,
        key_dim=32,
        value_dim=64,
        heads=8,
        norm_queries=True,
    ):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {"padding": padding, "stride": stride}
        self.to_q = nn.Conv2d(
            chan, key_dim * heads, kernel_size, **conv_kwargs
        )
        self.to_k = nn.Conv2d(
            chan, key_dim * heads, kernel_size, **conv_kwargs
        )
        self.to_v = nn.Conv2d(
            chan, value_dim * heads, kernel_size, **conv_kwargs
        )

        out_conv_kwargs = {"padding": padding}
        self.to_out = nn.Conv2d(
            value_dim * heads, chan_out, kernel_size, **out_conv_kwargs
        )

    def forward(self, x, y=None, context=None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim**-0.25), (q, k))

        if context is not None:
            context = context.reshape(b, c, 1, -1)
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape(b, heads, k_dim, -1), (ck, cv))
            k = torch.cat((k, ck), dim=3)
            v = torch.cat((v, cv), dim=3)

        k = k.softmax(dim=-1)

        if self.norm_queries:
            q = q.softmax(dim=-2)

        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhdn,bhde->bhen", q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift
    # return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x**2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = m2 - m**2
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return (
            fused_bn(x, m, var, gain, bias, eps),
            m.squeeze(),
            var.squeeze(),
        )
    return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer("stored_mean", torch.zeros(num_channels))
        self.register_buffer("stored_var", torch.ones(num_channels))
        self.register_buffer("accumulation_counter", torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        # pylint: disable=no-member
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        # pylint: disable=no-member
        if self.training:
            out, mean, var = manual_bn(
                x, gain, bias, return_mean_var=True, eps=self.eps
            )
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = (
                    self.stored_mean * (1 - self.momentum)
                    + mean * self.momentum
                )
                self.stored_var[:] = (
                    self.stored_var * (1 - self.momentum)
                    + var * self.momentum
                )
            return out
        # If not in training mode, use the stored statistics
        mean = self.stored_mean.view(1, -1, 1, 1)
        var = self.stored_var.view(1, -1, 1, 1)
        # If using standing stats, divide them by the accumulation counter
        if self.accumulate_standing:
            mean = mean / self.accumulation_counter
            var = var / self.accumulation_counter
        return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if "ch" in norm_style:
        ch = int(norm_style.split("_")[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif "grp" in norm_style:
        groups = int(norm_style.split("_")[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        cross_replica=False,
        mybn=False,
        norm_style="bn",
    ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ["bn", "in"]:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # If using my batchnorm
        if self.mybn:
            return self.bn(x, gain=gain, bias=bias)
        if self.norm_style == "bn":
            out = F.batch_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "in":
            out = F.instance_norm(
                x,
                self.stored_mean,
                self.stored_var,
                None,
                None,
                self.training,
                0.1,
                self.eps,
            )
        elif self.norm_style == "gn":
            out = groupnorm(x, self.normstyle)
        elif self.norm_style == "nonorm":
            out = x
        return out * gain + bias

    def extra_repr(self):
        s = "out: {output_size}, in: {input_size},"
        s += " cross_replica={cross_replica}"
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(
        self,
        output_size,
        eps=1e-5,
        momentum=0.1,
        cross_replica=False,
        mybn=False,
    ):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn

        if mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        # Register buffers if neither of the above
        else:
            self.register_buffer("stored_mean", torch.zeros(output_size))
            self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x, y=None):
        if self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        return F.batch_norm(
            x,
            self.stored_mean,
            self.stored_var,
            self.gain,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


# Generator blocks (for BigGAN)
class GBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=nn.Conv2d,
        which_bn=bn,
        activation=None,
        upsample=None,
    ):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(
                in_channels, out_channels, kernel_size=1, padding=0
            )
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=SNConv2d,
        wide=True,
        preactivation=False,
        activation=None,
        downsample=None,
    ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = (
            True if (in_channels != out_channels) or downsample else False
        )
        if self.learnable_sc:
            self.conv_sc = self.which_conv(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)
