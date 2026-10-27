
from torch.utils.data import DataLoader
import glob
import os
import sys
import random
import argparse
sys.path.append(os.path.dirname(os.getcwd()))

from src.utils.variable_utils import *
from src.utils.model_utils import *
from src.utils.data_utils import *
from src.utils.variable_utils import *
from src.utils.viz_utils import *
from src.data.dataloaders import SynDatasetLabel
from src.models.exo_classifier import ExoClassifier, get_arc_params
from tdpy.verbosity import print


def train_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--idx',type=str, default='classifier/b')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--epoch',type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', action='store_true', help='If true set scheduler')
    parser.add_argument('--apply_lowpass', action='store_true', help='If true apply low pass filter to the input images')
    parser.add_argument('--sanity_checks', action='store_true')
    parser.add_argument('--train_pids', type=str, default='1386', help='Choose the pids to train the model on (use * for all programs)')
    parser.add_argument('--train_filters', type=str, default='*', help='Choose the filters to train the model on (f300m, f277w, f356w, f444w) (use * for all filters)')
    parser.add_argument('--mode', type=str, default='train', help='Choose the mode (train, test)')
    args = parser.parse_args()
    return args

def get_paths(args):

    train_pids = args.train_pids.split()
    train_filters = args.train_filters.split()
    mode = args.mode
    if not train_pids or not train_filters:
        raise ValueError("At least one proposal ID and filter are required")

    def find_paths(pattern, limit=None):
        print(f"Reading from {pattern}...")
        paths = glob.glob(pattern)
        return paths if limit is None else paths[:limit]

    injected = []
    not_injected = []
    if len(train_pids) == 1 and len(train_filters) == 1:
        proposal_id = train_pids[0]
        filter_name = train_filters[0]
        prefix = os.path.join(NIRCAM_DATA, f"{mode}/{proposal_id}/injections")
        injected = find_paths(os.path.join(prefix, f"*{filter_name}*fc*.npy"), 30000)
        not_injected = find_paths(os.path.join(prefix, f"*{filter_name}*[!fc].npy"), 55000)
        not_injected = list(set(not_injected) - set(injected))
    elif len(train_pids) == 1 or len(train_filters) == 1:
        for proposal_id in train_pids:
            for filter_name in train_filters:
                prefix = os.path.join(NIRCAM_DATA, f"{mode}/{proposal_id}/injections")
                injected.extend(
                    find_paths(os.path.join(prefix, f"*{filter_name}*fc*.npy"), 30000)
                )
                not_injected.extend(
                    find_paths(os.path.join(prefix, f"*{filter_name}*[!fc].npy"), 60000)
                )
        not_injected = list(set(not_injected) - set(injected))
    else:
        for proposal_id in train_pids:
            for filter_name in train_filters:
                prefix = os.path.join(NIRCAM_DATA, f"{mode}/{proposal_id}/injections")
                injected.extend(find_paths(os.path.join(prefix, f"*{filter_name}*fc*.npy")))
                not_injected.extend(find_paths(os.path.join(prefix, f"*{filter_name}*[!fc].npy")))

    print("INJECTED:",len(injected))
    print("NOT INJECTED:",len(not_injected))

    paths = injected + not_injected
    random.shuffle(paths)
    print("#Samples:",len(paths))

    return paths


def main():
    args = train_arg_parser()
    init_wandb(args)
    train_paths = get_paths(args=args)

    syndata = SynDatasetLabel(image_paths=train_paths, args=args)
    syndata_loader = DataLoader(
        dataset=syndata, batch_size=args.batch_size, shuffle=True
    )
    convdim_outputs, kernels_enc, strides_enc = get_arc_params()
    model = ExoClassifier(
        args=args,
        in_channels=1,
        latent_dim=32,
        convdim_enc_outputs=convdim_outputs,
        kernels=kernels_enc,
        strides=strides_enc,
    )
    model.train_model(model=model, train_dataloader=syndata_loader)


if __name__ == "__main__":
    main()