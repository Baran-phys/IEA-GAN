import argparse
import errno
import hashlib
import json
import os
import pathlib
import secrets
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.init


# Add IEA-GAN to path so that model can be imported
sys.path.append(f"{pathlib.Path(__file__).parent.parent.absolute()}")

import model as md  # pylint: disable=import-error,wrong-import-position


parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str, help="output directory")
parser.add_argument('checkpoint_file', type=str, help="trained model weights" )
parser.add_argument('num_events', type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num-producers', default=1, type=int)
parser.add_argument('--queue-maxsize', default=100, type=int)





def trunc_trick(bs, z_dim, bound= 0.8):
    z = torch.randn(bs, z_dim)
    while z.min() < -bound or bound < z.max():
        z = z.where((-bound < z) & (z < bound), torch.randn_like(z))
    return z



def load(model, checkpoint_file, device):
    state_dict = None
    try:
        checkpoint = torch.load(os.path.expandvars(checkpoint_file), map_location= device)
        model.to(device)
        model.load_state_dict(checkpoint)
    finally:
        del checkpoint
    return model


def thresh(x):
    return F.threshold(x,-0.26,-1)

def cut(img):
    return np.where(img > -0.26, img, -1)



def generate(model, batch_size=40, latent_dim= 128, num_classes= 40):
    with torch.no_grad():
        #latents = trunc_trick(batch_size, latent_dim, bound= 0.8).to(device)
        latents = torch.randn(batch_size, latent_dim, device='cuda')
        labels = torch.tensor([c for c in range(num_classes)], dtype=torch.long, device='cuda')
        imgs = model(latents, labels).detach().cpu()
        #Cut the noise below 7 ADU
        imgs = thresh(imgs)
        # center range [-1, 1] to [0, 1]
        imgs = imgs.mul_(0.5).add_(0.5)
        # renormalize and convert to uint8
        imgs = torch.pow(256, imgs).add_(-1).clamp_(0, 255).to(torch.uint8)
        # flatten channel dimension and crop 256 to 250
        imgs= imgs[:, 0, 3:-3, :]
        # find nonzero indices
        nonzeros = imgs.nonzero(as_tuple=True)
        # return combination of nonzero indices and corresponding values
        return tuple(t.tolist() for t in nonzeros), imgs[nonzeros].tolist()

def producer(checkpoint_file, device, queue):
    gen = md.Generator(**{'skip_init': True, 'no_optim': True, 'resolution': 256,
                     'D_attn' : '0', 'G_attn':'0', 'attn_type':'sa','n_head_G':2,
                     'G_ch' : 32, 'D_ch' : 32, 'relational_embed' : True,
                     'dim_z': 128, 'H_base': 3, 'G_shared': True, 'device': device,
                     'shared_dim': 128, 'hier': True, 'prior_embed': False}).to(device, torch.float32)
    model= load(gen, checkpoint_file, device)
    while True:
        queue.put(generate(model))

def run(output_file, num_events, queue, seed=None):
    import basf2  # pylint: disable=import-error
    from ROOT import Belle2  # pylint: disable=import-error

    VXDID = tuple(Belle2.VxdID(i+1, j+1, k+1) for i in range(2) for j in range(12 if i else 8) for k in range(2))

    class DigitCreator(basf2.Module):

        def initialize(self):
            self.digits = Belle2.PyStoreArray('PXDDigits')
            self.digits.registerInDataStore()

        def event(self):
            digits = self.digits
            nonzeros, charges  = queue.get()
            for idx, ucell, vcell, charge in zip(*nonzeros, charges):
                digit = digits.appendNew()
                digit.setSensorID(VXDID[idx])
                digit.setUCellID(ucell)
                digit.setVCellID(vcell)
                digit.setCharge(charge)
            del nonzeros, charges

    if seed is None:
        seed = secrets.randbelow(2**32 - 1)
    basf2.set_random_seed(seed)
    main = basf2.create_path()
    main.add_module('EventInfoSetter', evtNumList=[num_events])
    main.add_module('Progress')
    main.add_module(DigitCreator())
    main.add_module('RootOutput', outputFileName=output_file, updateFileCatalog=False)
    basf2.process(main)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()

    # validate num events
    if args.num_events <= 0:
        parser.error(f'expecting num events > 0 (got: {args.num_events})')

    # validate queue maxsize
    if args.queue_maxsize <= 0:
        parser.error(f'expecting queue maxsize > 0 (got: {args.queue_maxsize}')

    # validate num producers
    if args.num_producers <= 0:
        parser.error(f'expecting num producers > 0 (got: {args.num_producers})')

    # validate device specifier
    device = None
    try:
        device = str(torch.device(args.device))
    except RuntimeError:
        pass
    if device is None:
        parser.error(f'invalid device specifier {device!r}')

    # validate checkpoint file
    args.checkpoint_file = os.path.expandvars(args.checkpoint_file)
    if not (os.path.exists(args.checkpoint_file) and os.path.isfile(args.checkpoint_file)):
        parser.error(f'no checkpoint file found in {args.checkpoint_file!r}')

    # validate output file
    args.output_file = os.path.expandvars(args.output_file)
    if os.path.exists(args.output_file):
        parser.error(f'cannot overwrite existing file in {args.output_file!r}')
    else:
        try:
            with open(args.output_file, 'w') as f:
                pass
            os.remove(args.output_file)
        except OSError as e:
            if e.errno == errno.ENOENT:
                os.makedirs(os.path.dirname(args.output_file))

    # configure multiprocessing
    if 'cuda' in device:
        mp.set_start_method('spawn')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # compute hash of checkpoint
    with open(args.checkpoint_file, 'rb') as f:
        hash_func = hashlib.sha256()
        hash_func.update(f.read())
        digest = hash_func.hexdigest()
        del hash_func

    # start producers
    proc, queue = [], mp.Queue(args.queue_maxsize)
    for _ in range(args.num_producers):
        p = mp.Process(target=producer, args=(args.checkpoint_file, device, queue), daemon=True)
        p.start()
        proc.append(p)

    try:
        with open(args.output_file + '.json', 'w') as f:
            json.dump({'sha256(checkpoint)': digest, **vars(args)}, f, indent=4, sort_keys=True)
        run(args.output_file, args.num_events, queue)
        for p in proc:
            p.terminate()
    except KeyboardInterrupt:
        for p in proc:
            p.kill()
