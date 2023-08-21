import matplotlib.pyplot as plt
import numpy as np
import torch





def get_model(model_dir, device):

    model = torch.load(model_dir, map_location=device)

    return model


def get_testset(test_dirs):

    with open(test_dirs) as file:
        lines = [line.rstrip() for line in file]

    print(lines)
    print(len(lines))


get_testset('/home/sarperyn/sarperyurtseven/ProjectFiles/scripts/text_dirs.txt')

