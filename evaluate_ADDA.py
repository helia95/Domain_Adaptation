import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
from torch.utils.data import DataLoader
import torch.utils.data
from dataset import DANN_test
import numpy as np
import ADDA_model
import csv
import sys
args = sys.argv

root_dir = args[1]
target_name = args[2]
csv_file = args[3]


root_f = 'mod/adda/{}/target_encoder.pth'.format(target_name)
root_c = 'mod/adda/{}/class_predictor.pth'.format(target_name)

check_f = torch.load(root_f)
check_c = torch.load(root_c)


img_size = 28

dataset = DANN_test(root=root_dir, img_size=img_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


feature_extractor = ADDA_model.feature_extractor()
class_classifier = ADDA_model.class_classifier()

feature_extractor.load_state_dict(check_f)
class_classifier.load_state_dict(check_c)

if torch.cuda.is_available():
    feature_extractor.cuda()
    class_classifier.cuda()

feature_extractor.eval()
class_classifier.eval()

with open(csv_file, 'a') as writeFile:
    for i, (img, img_name) in enumerate(dataloader):

        if torch.cuda.is_available():
            img = img.cuda()

        f = feature_extractor(img)
        pred = class_classifier(f)
        pred = pred.data.cpu()

        row = [img_name[0], np.argmax(pred.numpy())]
        writer = csv.writer(writeFile)
        writer.writerow(row)




