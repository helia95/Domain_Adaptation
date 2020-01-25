import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import DANN
import DANN_model
import numpy as np
from torch.autograd import Variable


training_mode = 'dann'
#training_mode = 'stand_alone'

source_name = 'usps'
target_name = 'mnistm'
img_size = 28
batch_size = 256
epochs = 100
learning_rate = 0.01

random.seed(999)
torch.manual_seed(999)

if training_mode == 'dann':

    model_root = 'dann_models/dann/{0}_to_{1}'.format(source_name, target_name)

    dataset_source = DANN(name=source_name, img_size=img_size, train=True)
    dataloader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_target = DANN(name=target_name, img_size=img_size, train=True)
    dataloader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

    if (target_name == 'svhn'):
        feature_extractor = DANN_model.svhn_feature_extractor()
        class_classifier = DANN_model.svhn_class_classifier()
        domain_classifier = DANN_model.svhn_domain_classifier()
    else:
        feature_extractor = DANN_model.feature_extractor()
        class_classifier = DANN_model.class_classifier()
        domain_classifier = DANN_model.domain_classifier()

    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                           {'params': class_classifier.parameters()},
                           {'params': domain_classifier.parameters()}],
                          lr=learning_rate, momentum=0.9)


    loss_cl = nn.NLLLoss()
    loss_dm = nn.NLLLoss()

    if torch.cuda.is_available():
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()
        loss_cl.cuda()
        loss_dm.cuda()

    for p in feature_extractor.parameters():
        p.requires_grad = True
    for p in class_classifier.parameters():
        p.requires_grad = True
    for p in domain_classifier.parameters():
        p.requires_grad = True

    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    len_data = min(len(dataloader_source), len(dataloader_target))

    for epoch in range(epochs):

        if epoch == 25:
            learning_rate = 0.001
        if epoch == 60:
            learning_rate = 0.0001
        if epoch == 85:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        for i, (sdata, tdata) in enumerate(zip(dataloader_source, dataloader_target)):
            p = float(i + epoch * len_data) / epochs / len_data
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Get data
            s_img, s_label = sdata
            t_img, t_label = tdata
            batch_size = min(len(s_label), len(t_label))

            s_img, s_label = s_img[0:batch_size, :, :, :], s_label[0:batch_size]
            t_img, t_label = t_img[0:batch_size, :, :, :], t_label[0:batch_size]



            # Train on source
            optimizer.zero_grad()

            s_dm_label = torch.zeros(batch_size).long()

            if torch.cuda.is_available():
                s_img = Variable(s_img).cuda()
                s_label = Variable(s_label).cuda()
                s_dm_label = Variable(s_dm_label).cuda()



            s_features = feature_extractor(s_img)
            s_cl = class_classifier(s_features)
            s_dm = domain_classifier(s_features, alpha)


            errS_cl = loss_cl(s_cl, s_label)
            errS_dm = loss_dm(s_dm, s_dm_label)

            # Train on target
            t_dm_label = torch.ones(batch_size).long()

            if torch.cuda.is_available():
                t_img = Variable(t_img).cuda()
                t_dm_label = Variable(t_dm_label).cuda()


            t_features = feature_extractor(t_img)
            t_dm = domain_classifier(t_features, alpha)

            errT_dm = loss_dm(t_dm, t_dm_label)


            error = errT_dm + errS_dm + errS_cl
            error.backward()
            optimizer.step()

            if i % 10 == 0:
                print('[{}/{} {}/{}]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    epoch, epochs, i, len_data , error.item(), errS_cl.item(),
                    errS_dm.item()+errT_dm.item()
                ))

        torch.save(feature_extractor.state_dict(), '{}/features_encoder.pth'.format(model_root))
        torch.save(class_classifier.state_dict(), '{}/class_classifier.pth'.format(model_root))
        torch.save(domain_classifier.state_dict(), '{}/domain_classifier.pth'.format(model_root))



if training_mode == 'stand_alone':

    model_root = 'dann_models/stand_alone/{}/'.format(source_name)

    dataset_source = DANN(name=source_name, img_size=img_size, train=True)
    dataloader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)

    if source_name == 'svhn':
        feature_extractor = DANN_model.svhn_feature_extractor()
        class_classifier = DANN_model.svhn_class_classifier()
    else:
        feature_extractor = DANN_model.feature_extractor()
        class_classifier = DANN_model.class_classifier()


    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                           {'params': class_classifier.parameters()}],
                          lr=learning_rate, momentum=0.9)

    loss_cl = nn.NLLLoss()

    if torch.cuda.is_available():
        feature_extractor.cuda()
        class_classifier.cuda()
        loss_cl.cuda()

    for p in feature_extractor.parameters():
        p.requires_grad = True
    for p in class_classifier.parameters():
        p.requires_grad = True

    feature_extractor.train()
    class_classifier.train()

    for epoch in range(epochs):
        
        if epoch == 25:
            learning_rate = 0.001
        if epoch == 60:
            learning_rate = 0.0001
        if epoch == 85:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        for i, (s_img, s_label) in enumerate(dataloader_source):

            batch_size = len(s_label)

            optimizer.zero_grad()

            if torch.cuda.is_available():
                s_img = Variable(s_img).cuda()
                s_label = Variable(s_label).cuda()

            s_features = feature_extractor(s_img)
            s_cl = class_classifier(s_features)

            error = loss_cl(s_cl, s_label)

            error.backward()

            optimizer.step()

            if i % 10 == 0:
                print('[{}/{}]\tLoss: {:.6f}'.format(epoch, epochs, error.item()))


    torch.save(feature_extractor.state_dict(), '{}features_encoder.pth'.format(model_root))
    torch.save(class_classifier.state_dict(), '{}class_classifier.pth'.format(model_root))



