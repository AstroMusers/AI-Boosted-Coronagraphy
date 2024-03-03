import numpy
import shutil
import os
import glob
import random
import copy


BASE_DIR = "/data/scratch/sarperyurtseven/dataset"
INSTRUME  = "NIRCAM"
PID      = "1386"
FC       = "fc5"

DATA_DIR = os.path.join(BASE_DIR,INSTRUME,PID,"injections",FC + "_train") 
test_dir = copy.deepcopy(DATA_DIR)

data = glob.glob(os.path.join(DATA_DIR,"*.npy"))
n_test = int(len(data)*0.05)
print("BEFORE TRAIN DATA:",len(data))


test_dir = test_dir.replace('train','test')
os.makedirs(test_dir,exist_ok=True)

test_data = random.choices(data,k=n_test)


for i in test_data:
    try:
        i_copy = copy.deepcopy(i)
        destination = i_copy.replace('train','test')
        os.rename(i,destination)
        os.remove(i)
    except:
        print("no directory")


print("AFTER TRAIN DATA:",len(data))
print("AFTER TEST DATA:",len(glob.glob(os.path.join(test_dir,"*.npy"))))


    

