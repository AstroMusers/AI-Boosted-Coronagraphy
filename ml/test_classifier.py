
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import glob
import os
import sys
import wandb
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
sys.path.append(os.path.dirname(os.getcwd()))
from scripts.visualization_helpers import *


from util.util_data import *
from util.util_dirs import *
from util.util_train import *
from models import ExoClassifier


### TEST classifier script


def inference_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--real_data',type=str, default='/data/scratch/sarperyurtseven/dataset/NIRCAM/')# /data/scratch/sarperyurtseven/results/training_results
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--model', type=str, default='ae')
    parser.add_argument('--inference_folder', type=str, default='fc5_test', help='Train folder name') # fc5_injections_test
    parser.add_argument('--apply_lowpass', action='store_true', help='If true apply low pass filter to the input images')
    parser.add_argument('--model_path', type=str, default="/data/scratch/sarperyurtseven/results/training_results/classifier/0/models/model.pt")
    parser.add_argument('--result_savepath', type=str, default='/data/scratch/sarperyurtseven/results/training_results/classifier_lp/0/results_fig')
    parser.add_argument('--syn', action='store_true')
    args = parser.parse_args()
    return args


def get_arc_params():

    kernels_enc = [7,5,5,3]
    paddings_enc= [1,1,1,1]
    strides_enc = [2,1,1,2]

    maxpool = [0,0,0,0,0,0,0]

    kernels_dec = list(reversed(kernels_enc))
    paddings_dec= [1,1,1,1]
    strides_dec = list(reversed(strides_enc))

    convdim_outputs = calculate_conv_dims(80,paddings_enc,kernels_enc,strides_enc,maxpool)

    convtrans_outputs = calculate_convtrans_dim(convdim_outputs[-1],paddings_dec,kernels_dec,strides_dec)

    return convdim_outputs, kernels_enc, strides_enc


def get_model(model_dir, device):



    convdim_outputs, kernels_enc, strides_enc = get_arc_params()

    model = ExoClassifier(args=args,
               in_channels=1,
               latent_dim=8,
               convdim_enc_outputs=convdim_outputs, 
               kernels=kernels_enc, 
               strides=strides_enc)
    
    model = torch.load(model_dir, map_location=device)
    
    return model

def get_testloader(args):


    if args.syn:
        print(os.path.join(args.real_data, args.inference_folder, '*.npy'))
        test_paths = glob.glob(os.path.join(args.real_data, args.inference_folder, '*.npy'))
    
    else:       
        print(os.path.join(INJECTIONS, args.inference_folder, '*fc5.npy'))
        inj = glob.glob(os.path.join(INJECTIONS, args.inference_folder, '*fc5.npy'))[:args.batch_size]
        no_inj = glob.glob(os.path.join(INJECTIONS, args.inference_folder, '*[!fc5].npy'))[:args.batch_size]

        print("INJ:",len(inj))
        print("No INJ:",len(no_inj))
        test_paths = inj + no_inj
        

    syndata        = SynDatasetLabel(image_paths=test_paths, args=args)
    syndata_loader = DataLoader(dataset=syndata, batch_size=1, shuffle=True)

    return syndata_loader, test_paths


def test_model(model, dataloader, args):
    test_losses = []
    model.eval()
    test_loss = 0
    correct = 0
    results = []

    with torch.no_grad():
        for image, target, filtered_image, img_path in dataloader:
            target = target.type(torch.LongTensor).to(args.device)

            if args.apply_lowpass:
                batch = filtered_image.to(args.device)
            else:
                batch = image.to(args.device)

            z = model.encode(batch)
            z = model.fc_layers(z)
            z = model.final_layer(z)
            output = F.log_softmax(z, dim=1)  # Specify the dimension for softmax

            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            results.append((img_path, pred, target))
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(dataloader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset), accuracy))

    return results


def plot_confusion_matrix(results, args, save_path):
    preds, targets = [], []

    # Extract predictions and targets from the 'results' list
    for i in range(len(results)):
        preds.append(results[i][1].detach().cpu().numpy())
        targets.append(results[i][2].detach().cpu().numpy())

    # Convert lists to NumPy arrays
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate the confusion matrix and F1 score
    cm = confusion_matrix(y_true=targets, y_pred=preds)
    tn, fp, fn, tp = confusion_matrix(y_true=targets, y_pred=preds).ravel()
    print(f"TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
    f1 = f1_score(y_true=targets, y_pred=preds)

    # Create a DataFrame for the confusion matrix
    df_cm = pd.DataFrame(cm, index=["Negative", "Positive"], columns=["Predicted Negative", "Predicted Positive"])

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')  # Use 'd' to format values as integers
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()

    # Save the plot with a file name based on 'args.model_path'
    model_name = '-'.join(args.model_path.split("/")[-4:-3])
    plt.savefig(os.path.join(save_path, f'{i}.jpg'), format='jpg', pad_inches=0, dpi=200)

    # Print the F1 score and confusion matrix
    print("F1 score:", f1)
    print("Confusion Matrix:")
    print(cm)


def get_batch(dirs):
    batch = []
    for i in dirs:
        data = np.load(i)
        batch.append(data)
    batch = np.concatenate(np.expand_dims(batch, axis=0))
    return batch


def get_exo_coords(image_path):
    idx_x  = image_path.rfind('x')

    if idx_x == -1:
        x, y = 0, 0
    else:
        x = int(image_path[idx_x+1:idx_x+3])
        y = int(image_path[idx_x+5:idx_x+7])

    return x,y

def get_batch_coord(coords_dict, idx,bs):

    xs = []
    ys = []
    for i in range(bs):

        xs.append(coords_dict[idx][i][0]) 
        ys.append(coords_dict[idx][i][1])

    return xs, ys


def visualize_results(results, model_name, save_path):
    results_dict = {
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

    for i in range(len(results)):
        if results[i][1].item() == results[i][2].item() and int(results[i][1]) == 1:
            results_dict['TP'].append(results[i][0][0])
            x,y = get_exo_coords(results[i][0][0])
            coords_dict['TP'].append((x,y))

        elif results[i][1] == results[i][2] and int(results[i][1]) == 0:
            results_dict['TN'].append(results[i][0][0])
            x,y = get_exo_coords(results[i][0][0])
            coords_dict['TN'].append((x,y))

        elif results[i][1] != results[i][2] and int(results[i][1]) == 1:
            results_dict['FP'].append(results[i][0][0])
            x,y = get_exo_coords(results[i][0][0])
            coords_dict['FP'].append((x,y))

        elif results[i][1] != results[i][2] and int(results[i][1]) == 0:
            results_dict['FN'].append(results[i][0][0])
            x,y = get_exo_coords(results[i][0][0])
            coords_dict['FN'].append((x,y))


    step = 20 * 0.06259530358142339
    step = round(step, 2)
    labels = step * np.array([-2., -1., 0., 1., 2.])
    axis_points = np.linspace(0, 80, 5)

    for i in ['TP', 'FP', 'TN', 'FN']:
        data = results_dict[i]
        
        print(i,len(data))
        if len(data) == 0:
            continue

        batch = get_batch(data)

        bs = min(4, len(data))

        x,y  = get_batch_coord(coords_dict,i,bs)

        fig, axes = plt.subplots(1, bs, figsize=(bs * 5.5, 10.5))

        if bs == 1:
            axes = [axes]  # Ensure 'axes' is a list even if there's only one subplot

        for col, ax in enumerate(axes):
            ax.imshow(batch[col], interpolation='nearest')

            if (x[col] != 0) and (y[col] != 0):
                ax.text(x[col], y[col], s="\u25CF", fontsize=10, color='red', alpha=.5, ha='center', va='center')

            if col == 0:
                ax.set_ylabel('arcsec', fontsize=10,)
                ax.set_xlabel('arcsec', fontsize=10,)
                ax.set_yticks(axis_points, labels, fontsize=10, rotation=0)

            else:
                ax.set_yticks([])
                ax.set_xticks([])
            ax.set_xticks(axis_points, labels, fontsize=10, rotation=75)

        plt.subplots_adjust(wspace=.25, hspace=0)
        plt.savefig(os.path.join(save_path, f'{i}.jpg'), format='jpg', bbox_inches='tight', pad_inches=.1, dpi=200)
        plt.close()



args = inference_arg_parser()

model_name = '-'.join(args.model_path.split("/")[-4:-2])
model = get_model(args.model_path, args.device)
dataloader, test_paths = get_testloader(args)
results = test_model(model, dataloader, args)

locations        = get_psf_info(test_paths)
info             = get_augmentation_info(test_paths)
transformed_list = do_transformations(info, locations)
#arrays           = get_array(test_paths)



save_path = os.path.join(args.result_savepath, model_name, "fc5")

if not os.path.exists(save_path):
    os.makedirs(save_path)

plot_confusion_matrix(results, args, save_path)
visualize_results(results, model_name, save_path=save_path)
