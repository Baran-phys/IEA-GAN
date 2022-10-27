''' train_fns.py
Functions for the main loop of training
'''
import torch

import utils
import loss

from mycleanfid import fid
from cr_diff_aug import CR_DiffAug

# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        # pylint: disable=unused-argument
        return {}
    return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config, device):
    # G_bs = max(config['G_batch_size'], config['batch_size'])  # commented, variable unused
    contra_criter = loss.Conditional_Contrastive_loss(device, config['batch_size'], config['pos_collected_numerator'])
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # How many chunks to split x and y into?
        if config['Con_reg']:
            x_aug = CR_DiffAug(x)
            x = torch.split(x, config['batch_size'])
            y = torch.split(y, config['batch_size'])
            x_aug = torch.split(x_aug, config['batch_size'])
        else:
            x_aug = None
            x = torch.split(x, config['batch_size'])
            y = torch.split(y, config['batch_size'])

        counter = 0


        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        #Setting tempreture for Contra
        if config['conditional_strategy'] == 'Contra':
            t = 1.0

        for _ in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            for _ in range(config['num_D_accumulations']):
                z_.sample_()
                #y_ = torch.randperm(40, device=device, requires_grad=False)
                if config['conditional_strategy'] == 'Proj':
                    if config['Con_reg']:
                        D_fake, D_real, D_real_aug = GD(z_[:config['batch_size']], y[counter],
                                                        x[counter], y[counter], x_aug[counter],
                                                        contra=False, train_G=False,
                                                        split_D=config['split_D'], diff_aug=config['diff_aug'])

                        #Compute components of D's loss, average them, and divide by the number of gradient accumulations
                        D_loss_real, D_loss_fake = loss.loss_hinge_dis(D_fake, D_real)
                        D_loss = D_loss_real + D_loss_fake
                        consistency_loss = loss.l2_loss(D_real, D_real_aug)
                        D_loss += config['cr_lambda']*consistency_loss
                        D_loss = D_loss / float(config['num_D_accumulations'])
                    else:
                        D_fake, D_real = GD(z_[:config['batch_size']], y[counter],
                                            x[counter], y[counter], x_aug=None,
                                            contra=False, train_G=False,
                                            split_D=config['split_D'], diff_aug=config['diff_aug'])

                        # Compute components of D's loss, average them, and divide by
                        # the number of gradient accumulations
                        D_loss_real, D_loss_fake = loss.loss_hinge_dis(D_fake, D_real)
                        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

                elif config['conditional_strategy'] == 'Contra':
                    if config['Con_reg']:
                        #(y_[:config['batch_size']] vs y[counter])
                        (cls_proxies_fake, cls_embed_fake, D_fake,
                         cls_proxies_real, cls_embed_real, D_real,
                         cls_embed_real_aug, D_real_aug) = GD(z_[:config['batch_size']], y[counter],
                                                              x[counter], y[counter], x_aug[counter],
                                                              contra=True, train_G=False,
                                                              split_D=config['split_D'],
                                                              diff_aug=config['diff_aug'])


                        # Compute components of D's loss, average them, and divide by
                        # the number of gradient accumulations
                        D_loss_real, D_loss_fake = loss.loss_hinge_dis(D_fake, D_real)
                        D_loss = D_loss_real + D_loss_fake
                        real_cls_mask = utils.make_mask(y[counter], config['n_classes'], device)
                        D_loss += config['contra_lambda']*contra_criter(cls_embed_real, cls_proxies_real,
                                                                        real_cls_mask, y[counter], t, 0)
                        cls_consistency_loss = loss.l2_loss(cls_embed_real, cls_embed_real_aug)
                        consistency_loss = loss.l2_loss(D_real, D_real_aug)
                        consistency_loss += cls_consistency_loss
                        D_loss += config['cr_lambda']*consistency_loss
                        D_loss = D_loss / float(config['num_D_accumulations'])
                    else:
                        #(y_[:config['batch_size']] vs y[counter])
                        (cls_proxies_fake, cls_embed_fake, D_fake,
                         cls_proxies_real, cls_embed_real, D_real) = GD(z_[:config['batch_size']], y[counter],
                                                                        x[counter], y[counter], x_aug=None,
                                                                        contra=True, train_G=False,
                                                                        split_D=config['split_D'],
                                                                        diff_aug=config['diff_aug'])


                        # Compute components of D's loss, average them, and divide by
                        # the number of gradient accumulations
                        D_loss_real, D_loss_fake = loss.loss_hinge_dis(D_fake, D_real)
                        D_loss = D_loss_real + D_loss_fake

                        real_cls_mask = utils.make_mask(y[counter], config['n_classes'], device)
                        D_loss += config['contra_lambda']*contra_criter(cls_embed_real, cls_proxies_real,
                                                                        real_cls_mask, y[counter], t, 0)


                        if config['Uniformity_loss']:
                            unif_loss_d = loss.unif_loss(cls_embed_real)
                            D_loss += config['unif_lambda']*unif_loss_d

                        D_loss = D_loss / float(config['num_D_accumulations'])


                D_loss.backward()

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                utils.ortho(D, config['D_ortho'])

            if config['clip_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(D.parameters(), config['clip_norm'])

            D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        # If accumulating gradients, loop multiple times
        for _ in range(config['num_G_accumulations']):
            z_.sample_()
            #y_ = torch.randperm(40, device=device, requires_grad=False)
            if config['conditional_strategy'] == 'Proj':
                D_fake = GD(z_, y_, x_aug=None, contra=False,
                            train_G=True, split_D=config['split_D'],
                            diff_aug=config['diff_aug'])

                G_loss = loss.loss_hinge_gen(D_fake)/float(config['num_G_accumulations'])

            elif config['conditional_strategy'] == 'Contra':
                (cls_proxies_fake, cls_embed_fake, D_fake)= GD(z_, y[counter], x_aug=None,
                                                                contra=True, train_G=True,
                                                                split_D=config['split_D'],
                                                                diff_aug=config['diff_aug'])

                fake_cls_mask = utils.make_mask(y[counter], config['n_classes'], device)
                G_loss = loss.loss_hinge_gen(D_fake)
                G_loss += config['contra_lambda']*contra_criter(cls_embed_fake, cls_proxies_fake,
                                                                fake_cls_mask, y[counter], t, 0)

                if config['IEA_loss']:
                    iea_loss = loss.IEA_loss(cls_embed_fake, cls_embed_real)
                    G_loss += config['IEA_lambda']*iea_loss


                    if config['Uniformity_loss']:
                        unif_loss_g = loss.unif_loss(cls_embed_fake)
                        G_loss += config['unif_lambda']*unif_loss_g

                G_loss = G_loss/float(config['num_G_accumulations'])
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
            blacklist=[param for param in G.shared.parameters()])

        if config['clip_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(G.parameters(), config['clip_norm'])
            G.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item()),
               'unif_loss_d': float(unif_loss_d.item()),
               'iea_loss': float(iea_loss.item())}

        # Return the components of G's and D's loss.
        return out
    return train


def test(G, D, G_ema, state_dict, config, test_log):
    """
    This function runs the inception metrics code, checks if the results
    are an improvement over the previous best FID and logs the results
    """
    print('Gathering inception metrics...')
    # y_ = torch.randperm(40, device=config["device"], requires_grad=False)  # commented, variable unused
    FID = fid.compute_fid(gen=G, dataset_name="pxd_sim_test_com", dataset_res=256,
                            batch_size=40, mode="clean", dataset_split="custom",
                            z_dim=128, num_gen=config['num_incep_images'], trunc=None,
                            device=config["device"])
    print(f"The FID score is {FID}")
    # If improved over previous best metric, save approrpiate copy state_dict['itr']
    if (config['which_best'] == 'FID' and FID < state_dict['best_FID']):
        print('%s improved over previous best, saving checkpoint...' % config['which_best'])
        #save_weights(G, D, state_dict, config['weights_root'],
                     #experiment_name, 'best%d' % state_dict['save_best_num'],
                     #G_ema if config['ema'] else None)
        #save_weights(G, D, state_dict, config['weights_root'],
                     #experiment_name, 'best%d' % state_dict['itr'],
                     #G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), FID=float(FID))
