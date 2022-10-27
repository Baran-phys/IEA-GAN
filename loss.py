import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd



def unif_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()




def IEA_loss(k_f, k_r):
    with torch.no_grad():
        attn_logits_r = torch.matmul(k_r, k_r.transpose(-2, -1))
        attention_r = F.softmax(attn_logits_r, dim=-1)

    attn_logits_f = torch.matmul(k_f, k_f.transpose(-2, -1))
    attention_f = F.log_softmax(attn_logits_f, dim=-1)

    kld = torch.nn.KLDivLoss(reduction="batchmean")
    loss = kld(attention_f, attention_r)
    # loss = F.smooth_l1_loss(attn_logits_f, attn_logits_r, reduction='mean', beta= 0.5)
    # mse = torch.nn.MSELoss()
    # loss = mse(attn_logits_f, attn_logits_r)
    return loss


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1.0 - dis_real))
    loss_fake = torch.mean(F.relu(1.0 + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def l2_loss(dis_real, dis_aug_real):
    mse = torch.nn.MSELoss()
    loss = mse(dis_real, dis_aug_real)
    return loss


def set_temperature(
    conditional_strategy,
    tempering_type,
    start_temperature,
    end_temperature,
    step_count,
    tempering_step,
    total_step,
):
    if conditional_strategy == "Contra":
        if tempering_type == "continuous":
            t = (
                start_temperature
                + step_count
                * (end_temperature - start_temperature)
                / total_step
            )
        elif tempering_type == "discrete":
            tempering_interval = total_step // (tempering_step + 1)
            t = (
                start_temperature
                + (step_count // tempering_interval)
                * (end_temperature - start_temperature)
                / tempering_step
            )
        else:
            t = start_temperature
    else:
        t = "no"
    return t


class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(
        self, inst_embed, proxy, negative_mask, labels, temperature, margin
    ):
        similarity_matrix = self.calculate_similarity_matrix(
            inst_embed, inst_embed
        )
        instance_zone = torch.exp(
            (self.remove_diag(similarity_matrix) - margin) / temperature
        )

        inst2proxy_positive = torch.exp(
            (self.cosine_similarity(inst_embed, proxy) - margin) / temperature
        )
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(
                mask_4_remove_negatives
            )
            inst2inst_positives = instance_zone * mask_4_remove_negatives

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive

        denomerator = torch.cat(
            [torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone],
            dim=1,
        ).sum(dim=1)
        criterion = -torch.log(temperature * (numerator / denomerator)).mean()
        return criterion


class Conditional_Contrastive_loss_plus(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss_plus, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(
        self, inst_embed, proxy, negative_mask, labels, temperature, margin
    ):
        p2i_similarity_matrix = self.calculate_similarity_matrix(
            proxy, inst_embed
        )
        i2i_similarity_matrix = self.calculate_similarity_matrix(
            inst_embed, inst_embed
        )
        p2i_similarity_zone = torch.exp(
            (p2i_similarity_matrix - margin) / temperature
        )
        i2i_similarity_zone = torch.exp(
            (i2i_similarity_matrix - margin) / temperature
        )

        mask_4_remove_negatives = negative_mask[labels]
        p2i_positives = p2i_similarity_zone * mask_4_remove_negatives
        i2i_positives = i2i_similarity_zone * mask_4_remove_negatives

        p2i_numerator = p2i_positives.sum(dim=1)
        i2i_numerator = i2i_positives.sum(dim=1)
        p2i_denomerator = p2i_similarity_zone.sum(dim=1)
        i2i_denomerator = i2i_similarity_zone.sum(dim=1)

        p2i_contra_loss = -torch.log(
            temperature * (p2i_numerator / p2i_denomerator)
        ).mean()
        i2i_contra_loss = -torch.log(
            temperature * (i2i_numerator / i2i_denomerator)
        ).mean()
        return p2i_contra_loss + i2i_contra_loss


def calc_derv4gp(
    netD, conditional_strategy, real_data, fake_data, real_labels, device
):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1)
    alpha = (
        alpha.expand(batch_size, real_data.nelement() // batch_size)
        .contiguous()
        .view(batch_size, c, h, w)
    )
    alpha = alpha.to(device)

    real_data = real_data.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy == "Contra":
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == "Proj":
        disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv4dra(netD, conditional_strategy, real_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_data = real_data.to(device)
    differences = (
        0.5 * real_data.std() * torch.rand(real_data.size()).to(device)
    )

    interpolates = real_data + (alpha * differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy == "Contra":
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == "Proj":
        disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv(inputs, labels, netD, conditional_strategy, device, netG=None):
    zs = autograd.Variable(inputs, requires_grad=True)
    fake_images = netG(zs, labels)

    if conditional_strategy == "Contra":
        _, _, dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy == "Proj":
        dis_out_fake = netD(fake_images, labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(
        outputs=dis_out_fake,
        inputs=zs,
        grad_outputs=torch.ones(dis_out_fake.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
    return gradients, gradients_norm
