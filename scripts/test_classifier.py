
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
import glob
import os
import sys
import wandb
import argparse
import re
import pandas as pd
import random
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryROC
from tdpy.verbosity import print
sys.path.append(os.path.dirname(os.getcwd()))
from scripts.visualization_helpers import *


from src.utils.variable_utils import *
from src.utils.model_utils import *
from src.utils.data_utils import *
from src.utils.variable_utils import *
from src.utils.viz_utils import *
from src.data.dataloaders import SynDatasetLabel
from src.models.exo_classifier import ExoClassifier, get_arc_params
from src.utils.query_utils import get_dataset_dir


def _save_figure(path, **kwargs):
    print(f"Writing to {path}...")
    plt.savefig(path, **kwargs)


def inference_arg_parser():

    parser = argparse.ArgumentParser()
    #parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    #parser.add_argument('--seed',type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--apply_lowpass', action='store_true', help='If true apply low pass filter to the input images')
    parser.add_argument('--train_pids', type=str, default='1386', help='Choose the pids to train the model on (use * for all programs)')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join(PLOT_SAVE_PATH, 'training_results', 'dospid', '2', 'models', 'model.pt'),
        help='Choose the model to test)',
    )
    parser.add_argument('--train_filters', type=str, default='*', help='Choose the filters to train the model on (f300m, f277w, f356w, f444w) (use * for all filters)')
    parser.add_argument('--mode', type=str, default='test', help='Choose the mode (train, test)')
    parser.add_argument('--sanity_checks', action='store_true')
    args = parser.parse_args()
    return args

def get_model(model_dir, device):

    convdim_outputs, kernels_enc, strides_enc = get_arc_params()

    model = ExoClassifier(args=args,
               in_channels=1,
               latent_dim=8,
               convdim_enc_outputs=convdim_outputs, 
               kernels=kernels_enc, 
               strides=strides_enc)
    
    print(f"Reading from {model_dir}...")
    model = torch.load(model_dir, map_location=device)
    
    return model

def plot_confusion_matrix(targets, preds, save_path):
    
    cm = confusion_matrix(y_true=targets, y_pred=preds)
    tn, fp, fn, tp = confusion_matrix(y_true=targets, y_pred=preds).ravel()
    print(f"TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
    f1 = f1_score(y_true=targets, y_pred=preds)

    df_cm = pd.DataFrame(cm, index=["Negative", "Positive"], columns=["Predicted Negative", "Predicted Positive"])

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()

    print("F1 score:", f1)
    print("Confusion Matrix:")
    print(cm)
    _save_figure(os.path.join(save_path, 'confusion_matrix.jpg'), format='jpg', pad_inches=0, dpi=200)

def plot_precision_recall_curve(targets, probs, save_path):

    metric = BinaryPrecisionRecallCurve()
    metric.update(preds=probs, target=targets)
    fig, ax = metric.plot(score=True)
    _save_figure(os.path.join(save_path, 'precision_recall_curve.jpg'))

def plot_roc_curve(targets, probs, save_path):

    metric = BinaryROC()
    metric.update(preds=probs, target=targets)
    fig_, ax_ = metric.plot(score=True)
    _save_figure(os.path.join(save_path, 'roc_curve.jpg'))

def get_test_paths(args):
    return get_paths(args)


def _glob_paths(pattern):
    print(f"Reading from {pattern}...")
    return glob.glob(pattern)


def get_paths(args):

    train_pids = args.train_pids.split(' ')
    train_filters = args.train_filters.split(' ')
    mode = args.mode

    if len(train_pids) == 1:
        train_pids = train_pids[0]

        if len(train_filters) == 1:
            train_filters = train_filters[0]
            injected     = _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{train_filters}*fc*.npy'))
            not_injected = _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{train_filters}*[!fc].npy'))
            not_injected = list(set(not_injected) - set(injected))

        else:
            injected = []
            not_injected = []

            for f in train_filters:
                injected     += _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{f}*fc*.npy'))
                not_injected += _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{train_pids}/injections/*{f}*[!fc].npy'))
                not_injected = list(set(not_injected) - set(injected))

    else:
        injected = []
        not_injected = []

        if len(train_filters) == 1:
            for pid in train_pids:
                injected     += _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{train_filters}*fc*.npy'))
                not_injected += _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{train_filters}*[!fc].npy'))

        else:
            for pid in train_pids:
                for f in train_filters:
                    injected     += _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{f}*fc*.npy'))
                    not_injected += _glob_paths(os.path.join(NIRCAM_DATA,f'{mode}/{pid}/injections/*{f}*[!fc].npy'))

    print("INJECTED:",len(injected))
    print("NOT INJECTED:",len(not_injected))

    paths = injected + not_injected
    random.shuffle(paths)
    print("#Samples:",len(paths))

    return paths

def get_testloader(args):

    #test_paths = get_paths(args)        
    test_paths = get_test_paths(args)

    syndata        = SynDatasetLabel(image_paths=test_paths, args=args)
    syndata_loader = DataLoader(dataset=syndata, batch_size=1, shuffle=True)

    return syndata_loader, test_paths

def get_batch(dirs, sanity_checks=False):
    batch = []
    for i in dirs:
        print(f"Reading from {i}...")
        data = np.load(i)
        if sanity_checks:
            if data.ndim != 2:
                raise ValueError("Evaluation image must be a two-dimensional array")
            if not np.isfinite(data).all():
                raise ValueError("Evaluation image must contain only finite values")
            if data.size == 0 or np.ptp(data) == 0:
                raise ValueError("Evaluation image must have a nonzero range")
        batch.append(data)
    if sanity_checks and not batch:
        raise ValueError("Evaluation batch must contain at least one image")
    batch = np.concatenate(np.expand_dims(batch, axis=0))
    return batch

def get_exo_coords(image_path):
    idx_x  = image_path.rfind('x')

    if idx_x == -1:
        x, y = 0, 0  # [pixel]
    else:
        x = int(image_path[idx_x+1:idx_x+3])  # [pixel]
        y = int(image_path[idx_x+5:idx_x+7])  # [pixel]

    return x,y

def get_batch_coord(coords_dict, idx, bs):

    xs = []
    ys = []
    for i in range(bs):

        xs.append(coords_dict[idx][i][0]) 
        ys.append(coords_dict[idx][i][1])

    return xs, ys

def get_psf_info_bs1(injection_dir, pid):

    root_dir = os.path.join(
        get_dataset_dir(), "NIRCAM", "train", pid, "mastDownload", "JWST"
    )
    sew = set()
    star_location_info = []        
    psf_name = '-'.join(injection_dir.split('/')[-1].split('-')[:4]) + '_psfstack.fits'
    complete_dirs = os.path.join(root_dir, psf_name)
    print(f"Reading from {complete_dirs}...")
    with fits.open(complete_dirs) as psf_stack:
        wcs = get_wcs(psf_stack)
        ra, dec = get_ra_dec(psf_stack)
    sew.add((ra,dec))
    sky_coord = get_skycoord(ra, dec)
    x, y = skycoord_to_pixel(wcs, sky_coord)  # [pixel]
    star_location_info.append((x,y))
        
    return star_location_info

def get_arcsec_per_pixel(psf_path):
    print(f"Reading from {psf_path}...")
    with fits.open(psf_path) as psf_stack:
        return np.sqrt(psf_stack[1].header['PIXAR_A2'])  # [arcsec / pixel]

def get_augmentation_info1(img_path):

    numeric_info = []
    for lst in img_path.split('/')[-1].split('-')[6:-3]:

        numeric_info.append(re.findall(r'\d+', lst)[0])

    return numeric_info

def do_transformations(info, locations):

    transformed_list = []
    
    if len(info) == 6:

        y = 54-4#int(locations[idx][0]) 
        x = 36-5#int(locations[idx][1])

        rotate     = int(info[0])
        flip       = int(info[1])
        vertical   = int(info[2])
        horizontal = int(info[3])
        vshift     = int(info[4])
        hshift     = int(info[5])

        x, y = rotate_point(x, y, rotate*90)

        x, y = flip_point(x, y, flipud=True if flip == 1 or flip == 3 else False, fliplr=True if flip == 2 or flip == 3 else False)
        x, y = find_new_coordinates_after_shift(x, y, right_shift=hshift if horizontal == 2 else -hshift, down_shift=vshift if vertical == 2 else -vshift)

        transformed_list.append((int(x), int(y)))

    else:
        y = 54#int(locations[idx][0]) 
        x = 36#int(locations[idx][1])
        transformed_list.append((y, x))

    return transformed_list

def calculate_distance(x1, y1, x2, y2):
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_related_confusion_entries(preds_dict, coords_dict, star_coords_dict, probs_dict, pred, prob, target, img_path):


    locations        = get_psf_info_bs1(img_path,'1386') 
    info             = get_augmentation_info1(img_path)
    transformed_list = do_transformations(info, locations)
    transformed_star = transformed_list[0]


    if pred == target and int(pred) == 1:
        preds_dict['TP'].append(img_path)
        x,y = get_exo_coords(img_path)
        coords_dict['TP'].append((x,y))
        star_coords_dict['TP'].append(transformed_star) 
        probs_dict['TP'].append(prob)

    elif pred == target and int(pred) == 0:
        preds_dict['TN'].append(img_path)
        x,y = get_exo_coords(img_path)
        coords_dict['TN'].append((x,y))
        star_coords_dict['TN'].append(transformed_star)
        probs_dict['TN'].append(prob)

    elif pred != target and int(pred) == 1:
        preds_dict['FP'].append(img_path)
        x,y = get_exo_coords(img_path)
        coords_dict['FP'].append((x,y))
        star_coords_dict['FP'].append(transformed_star)
        probs_dict['FP'].append(prob)

    elif pred != target and int(pred) == 0:
        preds_dict['FN'].append(img_path)
        x,y = get_exo_coords(img_path)
        coords_dict['FN'].append((x,y))
        star_coords_dict['FN'].append(transformed_star)
        probs_dict['FN'].append(prob)

def validate_classifier_outputs(logits, probabilities, enabled=False):
    if not enabled:
        return
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != 2:
        raise ValueError("Classifier logits must have shape (batch, 2)")
    if probabilities.shape != (logits.shape[0], 1):
        raise ValueError("Classifier probabilities must have shape (batch, 1)")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Classifier logits must be finite")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Classifier probabilities must be finite")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Classifier probabilities must lie in [0, 1]")


def validate_classifier_targets(targets, batch_size, enabled=False):
    if not enabled:
        return
    if targets.ndim != 1 or targets.shape[0] != batch_size:
        raise ValueError("Classifier targets must match the batch size")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Classifier targets must be finite")
    if targets.is_floating_point() and not bool((targets == targets.round()).all()):
        raise ValueError("Classifier targets must be integer class indices")
    if bool(((targets < 0) | (targets >= 2)).any()):
        raise ValueError("Classifier targets must be binary class indices")


def test_model(model, dataloader, args):

    #results = []
    model.eval()
    preds   = [] 
    probs   = []
    targets = []
    paths   = []

    preds_dict = {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': []
    }

    probs_dict = {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': []
    }

    coords_dict = {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': []
    }

    star_coords_dict = {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': []
    }

    plot_save_path = '/'.join(args.model_path.split('/')[:-2])

    with torch.no_grad():
        for image, target, filtered_image, img_path in dataloader:
            if args.apply_lowpass:
                batch = filtered_image.to(args.device)
            else:
                batch = image.to(args.device)
            validate_classifier_targets(
                target, batch.shape[0], enabled=getattr(args, "sanity_checks", False)
            )
            target = target.type(torch.LongTensor).to(args.device)

            z = model.encode(batch)
            z = model.fc_layers(z)
            z = model.final_layer(z)
            output = F.log_softmax(z, dim=1) 

            pred = output.data.max(1, keepdim=True)[1]
            prob = output.data.max(1, keepdim=True)[0]
            probability = torch.exp(prob)
            validate_classifier_outputs(
                z, probability, enabled=getattr(args, "sanity_checks", False)
            )

            get_related_confusion_entries(preds_dict, coords_dict, star_coords_dict, probs_dict,  pred, prob, target, img_path[0])
            #results.append((img_path, pred, target))

            probs.append(probability)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            #paths.append(img_path)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    plot_confusion_matrix(targets, preds, plot_save_path)

    targets = torch.Tensor(targets).type(torch.LongTensor)
    probs = torch.Tensor(probs).type(torch.FloatTensor)

    plot_precision_recall_curve(targets, probs, plot_save_path)
    plot_roc_curve(targets, probs, plot_save_path)
 
    return preds_dict, coords_dict, star_coords_dict, probs_dict

def visualize_results(
    results_dict, coords_dict, star_coords_dict, sanity_checks=False
):
    

    step = 20 * 0.06259530358142339  # [arcsec]
    step = round(step, 2)  # [arcsec]
    labels = step * np.array([-2., -1., 0., 1., 2.])  # [arcsec]
    axis_points = np.linspace(0, 80, 5)  # [pixel]

    save_path = '/'.join(args.model_path.split('/')[:-2])
    plot_save_path = os.path.join(save_path, 'confusion_plots')
    os.makedirs(plot_save_path, exist_ok=True)

    for i in ['TP', 'FP', 'TN', 'FN']:
        data = results_dict[i]
        
        #print(i,len(data))
        if len(data) == 0:
            continue

        batch = get_batch(data, sanity_checks=sanity_checks)

        bs = min(4, len(data))

        x,y  = get_batch_coord(coords_dict,i,bs)

        fig, axes = plt.subplots(1, bs, figsize=(bs * 5.5, 10.5))

        if bs == 1:
            axes = [axes]  # Ensure 'axes' is a list even if there's only one subplot

        for col, ax in enumerate(axes):
            ax.imshow(batch[col], interpolation='nearest', cmap='Greys_r')
            
            if (x[col] != 0) and (y[col] != 0):
                ax.text(x[col], y[col], s="\u25CF", fontsize=10, color='red', alpha=.5, ha='center', va='center')
            ax.invert_yaxis()
            

            # if int(star_coords_dict[i][col][0]) == -1:
            #     pass
            # else:
            #print(star_coords_dict[i][col][0], star_coords_dict[i][col][1])
            ax.text(int(star_coords_dict[i][col][0]), int(star_coords_dict[i][col][1]) , s="\u2605", fontsize=30, color='green', ha='center', va='center')
    
            if col == 0:
                ax.set_ylabel('arcsec', fontsize=10,)
                ax.set_xlabel('arcsec', fontsize=10,)
                ax.set_yticks(axis_points, labels, fontsize=10, rotation=0)

            else:
                ax.set_yticks([])
                ax.set_xticks([])
            ax.set_xticks(axis_points, labels, fontsize=10, rotation=75)

        plt.subplots_adjust(wspace=.25, hspace=0)
        _save_figure(os.path.join(plot_save_path, f'{i}.jpg'), format='jpg', bbox_inches='tight', pad_inches=.1, dpi=200)
        plt.close()

def colormap_auc_angular_distance(preds_dict, coords_dict, star_coords_dict, probs_dict, arcsec_per_pix):
    
    ultimate_dict = get_angular_distances(preds_dict, coords_dict, star_coords_dict, probs_dict)
    auc_scores    = get_auc_scores(ultimate_dict)
    avg_angular_distances = [arcsec_per_pix * np.mean(np.array(expected)) for expected in [ultimate_dict['bin_distances'][key] for key in ultimate_dict['bin_distances']]]
    auc_scores = np.array(auc_scores)
    save_path = '/'.join(args.model_path.split('/')[:-2])

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_angular_distances, auc_scores, c=auc_scores, cmap='viridis')
    plt.colorbar(label='AUC')
    plt.xlabel('Angular Distance (arcsec)')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title('AUC as a Function of Angular Distance')
    _save_figure(os.path.join(save_path, 'colormap_plot_auc_angular_distance.jpg'), format='jpg', bbox_inches='tight', pad_inches=.1, dpi=200)
    plt.close()

def get_angular_distances(preds_dict, coords_dict, star_coords_dict, probs_dict):

    ultimate_dict = {

        'distances':[],
        'target':[],
        'probs':[],
        'img_paths':[],
    }
    
    for i in ['TP', 'FN']:
        for j in range(len(preds_dict[i])):
            x1, y1 = coords_dict[i][j]
            x2, y2 = star_coords_dict[i][j]

            distance = calculate_distance(x1, y1, x2, y2)  # [pixel]
            ultimate_dict['distances'].append(distance)
            ultimate_dict['target'].append(1 if i == 'TP' else 0)
            ultimate_dict['probs'].append(probs_dict[i][j][0][0].detach().cpu().numpy())
            #ultimate_dict['fluxes'].append(int(re.findall(r'fc\d+', preds_dict[i][j])[0][2:]))
            ultimate_dict['img_paths'].append(preds_dict[i][j])

    return ultimate_dict

def get_auc_scores(ultimate_dict):

    auc_scores = []

    for key in ultimate_dict['target']:
        targets = np.array([i for i in ultimate_dict['target'][key]])
        probs = np.array([j.detach().cpu().numpy() for j in ultimate_dict['probs'][key]])
        fpr, tpr, thresholds = roc_curve(targets, probs)
        auc_scores.append(auc(fpr, tpr))

    for i in auc_scores:
        print(i)
    return auc_scores

def colormap_auc_flux(preds_dict, probs_dict):
    
    ultimate_dict = get_fluxes(preds_dict, probs_dict)
    auc_scores    = get_auc_scores(ultimate_dict)
    flux = [str(int(key[2:])) for key in ultimate_dict['target']]
    auc_scores = np.array(auc_scores)
    save_path = '/'.join(args.model_path.split('/')[:-2])

    plt.figure(figsize=(10, 6))
    plt.scatter(flux, auc_scores, c=auc_scores, cmap='viridis')
    plt.colorbar(label='AUC')
    plt.xlabel('Flux (1/flux)')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title('AUC as a Function of Flux')
    _save_figure(os.path.join(save_path, 'colormap_plot_auc_flux.jpg'), format='jpg', bbox_inches='tight', pad_inches=.1, dpi=200)
    plt.close()

def get_fluxes(preds_dict, probs_dict):

    ultimate_dict = {

        'img_paths_bin':{'fc5':[],
                         'fc100':[], 
                         'fc1000':[],
                         'fc10000':[]},
        'target':{'fc5':[],
                    'fc100':[], 
                    'fc1000':[],
                    'fc10000':[]},

        'probs':{'fc5':[],
                    'fc100':[], 
                    'fc1000':[],
                    'fc10000':[]},
    }
    
    for i in ['TP', 'FN']:
        for j in range(len(preds_dict[i])):
            fc = re.findall(r'fc\d+',preds_dict[i][j])[0]
            print(fc)
            if fc in ['fc5', 'fc100', 'fc1000', 'fc10000']:
                ultimate_dict['img_paths_bin'][fc].append(preds_dict[i][j])
                ultimate_dict['target'][fc].append(1 if i == 'TP' else 0)
                ultimate_dict['probs'][fc].append(probs_dict[i][j][0])
            else:
                continue
    
    return ultimate_dict

def extract_flux_coefficient(path):
    match = re.search(r'fc([\d\.eE\+\-]+)\.npy', path)
    return float(match.group(1)) if match else None



def main():
    global args

    args = inference_arg_parser()
    model = get_model(args.model_path, args.device)
    dataloader, test_paths = get_testloader(args)

    psf_directory = os.path.join(
        get_dataset_dir(), "NIRCAM", "train", args.train_pids, "mastDownload", "JWST"
    )
    psf_path = _glob_paths(os.path.join(psf_directory, "*psfstack.fits"))[0]
    arcsec_per_pixel = get_arcsec_per_pixel(psf_path)

    preds_dict, coords_dict, star_coords_dict, probs_dict = test_model(
        model, dataloader, args
    )
    visualize_results(
        preds_dict, coords_dict, star_coords_dict,
        sanity_checks=args.sanity_checks,
    )

    ultimate_dict = get_angular_distances(
        preds_dict, coords_dict, star_coords_dict, probs_dict
    )
    dataframe = pd.DataFrame(ultimate_dict)
    dataframe['flux_coefficients'] = dataframe['img_paths'].apply(
        extract_flux_coefficient
    )
    dataframe['probs'] = np.exp(dataframe['probs'])
    results_path = os.path.join(PLOT_SAVE_PATH, 'results_fixed_.csv')
    print(f"Writing to {results_path}...")
    dataframe.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
