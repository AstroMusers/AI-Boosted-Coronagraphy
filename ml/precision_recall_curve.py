
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
from torch import rand, randint
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryROC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
sys.path.append(os.path.dirname(os.getcwd()))
from scripts.visualization_helpers import *


from util.util_data import *
from util.util_dirs import *
from util.util_train import *
from models import ExoClassifier

def inference_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed',type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--inference_folder', type=str, default='fc5_test', help='Train folder name')
    parser.add_argument('--model_path', type=str, default="/data/scratch/sarperyurtseven/results/training_results/classifier/0/models/model.pt")
    parser.add_argument('--result_savepath', type=str, default='/data/scratch/sarperyurtseven/results/training_results/classifier/0')
    parser.add_argument('--syn', action='store_true')
    parser.add_argument('--apply_lowpass', action='store_true', help='If true apply low pass filter to the input images')
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
    result_probs = []

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
            probs = output.data.max(1, keepdim=True)[0]
            pred = output.data.max(1, keepdim=True)[1]
            result_probs.append(torch.exp(probs))
            results.append((img_path, pred, target))
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(dataloader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset), accuracy))

    return results, result_probs

args = inference_arg_parser()

model_name = '-'.join(args.model_path.split("/")[-4:-2])
model = get_model(args.model_path, args.device)
dataloader, test_paths = get_testloader(args)
results, result_probs = test_model(model, dataloader, args)


targets = []

for point in results:
    targets.append(point[2][0])

targets = torch.Tensor(targets).type(torch.LongTensor)
result_probs = torch.Tensor(result_probs).type(torch.FloatTensor)



metric = BinaryPrecisionRecallCurve()
metric.update(preds=result_probs, target=targets)
fig, ax = metric.plot(score=True)
plt.savefig(os.path.join(args.result_savepath,'precision_recall_curve.jpg'))

metric = BinaryROC()
metric.update(preds=result_probs, target=targets)
fig_, ax_ = metric.plot(score=True)
plt.savefig(os.path.join(args.result_savepath,'roc_curve.jpg'))