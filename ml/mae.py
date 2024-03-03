from transformers import ViTMAEConfig, ViTMAEModel, AutoImageProcessor
from PIL import Image
from astropy.io import fits
from glob import glob
from tqdm import tqdm
import numpy as np
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def read_crfints():
    # Read the CRF data
    path = '/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/mastDownload/JWST/'
    crf_paths = glob(path + '*crf*.fits')
    crfs = []
    for crf_path in tqdm(crf_paths):
        crf = fits.open(crf_path)[1].data
        if crfs == []:
            crfs = crf
        else:
            crfs = np.concatenate((crf, crfs), axis=0)

    crfs = np.expand_dims(crfs, axis=1)
    return crfs

def read_psfstack():
    # Read the PSF stack
    path = '/data/scratch/bariskurtkaya/dataset/NIRCAM/1386/mastDownload/JWST/'
    psf_paths = glob(path + '*psfstack*.fits')
    psfs = []
    for psf_path in tqdm(psf_paths):
        psf = fits.open(psf_path)[1].data
        if psfs == []:
            psfs = psf
        else:
            psfs = np.concatenate((psf, psfs), axis=0)

    psfs = np.expand_dims(psfs, axis=1)
    return psfs


def fit_kmeans(data, n_clusters=2):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans

def check_accuracy(crfs_count, kmeans_labels):
    kmeans_labels_crfs = kmeans_labels[:crfs_count]
    kmeans_labels_psfs = kmeans_labels[crfs_count:]

    km0 = np.count_nonzero(kmeans_labels_crfs == 0)
    km1 = np.count_nonzero(kmeans_labels_crfs == 1)

    print(f"Cluster 0: {km0} Cluster 1: {km1}")

    km2 = np.count_nonzero(kmeans_labels_psfs == 0)
    km3 = np.count_nonzero(kmeans_labels_psfs == 1)

    print(f"Cluster 2: {km2} Cluster 3: {km3}")

    acc = km0 + km3 if km0 + km3 > km1 + km2 else km1 + km2

    print(f"Accuracy: {acc / (len(kmeans_labels))}")

    return acc / (len(kmeans_labels))



def main(model, image_processor):
    crfs = np.array(read_crfints())
    psfs = np.array(read_psfstack())
    
    print(crfs.shape, psfs.shape)

    data = np.concatenate((crfs, psfs), axis=0)
    data = np.nan_to_num(data, nan=np.nanmean(data))
    data = np.repeat(data[:, :, :, :,], 3, axis=1)
    data = np.uint8(255*(data - np.min(data)) / (np.max(data) - np.min(data)))

    data = [Image.fromarray(datum, "RGB").resize((224, 224)) for datum in data]

    inputs = image_processor(data, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)

    np_out = outputs.last_hidden_state.detach().cpu().numpy()[:, 0:1 :,].reshape(-1, 768)
    print(np_out.shape)
    kmeans = fit_kmeans(np_out)
    print(kmeans.labels_)
    
    acc = check_accuracy(len(crfs), kmeans.labels_)

    return acc


if __name__ == "__main__":

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device)

    main(model, image_processor)