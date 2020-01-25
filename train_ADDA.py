import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.utils.data
from dataset import DANN
import ADDA_model
from torch.autograd import Variable

random.seed(999)
torch.manual_seed(999)

source_name = 'mnistm'
target_name = 'svhn'

f = True

img_size = 28
batch_size = 64
epochs1 = 100
epochs2 = 100

model_root = 'adda_models/{}_to_{}/'.format(source_name, target_name)

dataset_source = DANN(name=source_name, img_size=img_size, train=True)
dataloader_source = DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_target = DANN(name=target_name, img_size=img_size, train=True)
dataloader_target = DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)

learning_rate = 0.01


feature_extractor_S = ADDA_model.feature_extractor()
class_classifier_S = ADDA_model.class_classifier()


optimizer_source = SGD(feature_extractor_S.parameters(), lr=learning_rate, momentum=0.9)
loss_s = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    feature_extractor_S.cuda()
    class_classifier_S.cuda()
    loss_s.cuda()

if f == True:

    feature_extractor_S.train()
    class_classifier_S.train()

    # Pretrain source
    for epoch in range(epochs1):
        if epoch == 25:
            learning_rate = 0.001
        if epoch == 60:
            learning_rate = 0.0001
        if epoch == 85:
            learning_rate = 0.00001
        for param_group in optimizer_source.param_groups:
            param_group['lr'] = learning_rate

        for i, (s_img, s_label) in enumerate(dataloader_source):

            batch_size = len(s_label)

            optimizer_source.zero_grad()

            if torch.cuda.is_available():
                s_img = Variable(s_img).cuda()
                s_label = Variable(s_label).cuda()

            s_features = feature_extractor_S(s_img)
            s_cl = class_classifier_S(s_features)

            error = loss_s(s_cl, s_label)

            error.backward()

            optimizer_source.step()

            if i % 10 == 0:
                print('[{}/{}]\tLoss disc: {:.6f}'.format(epoch, epochs1, error.item()))
    torch.save(feature_extractor_S.state_dict(), '{}source_encoder.pth'.format(model_root))
    torch.save(class_classifier_S.state_dict(), '{}class_predictor.pth'.format(model_root))
else:
    path = torch.load('{}source_encoder.pth'.format(model_root))
    feature_extractor_S.load_state_dict(path)

######################################################################################
print('\n******************************  Train discrimiantor and targer encoder ****************************************\n ')
# Train discrimiantor and target encoder


feature_extractor_T = ADDA_model.feature_extractor()
discriminator = ADDA_model.discriminator()


feature_extractor_T.load_state_dict(feature_extractor_S.state_dict())


optimizer_discriminator = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_target = Adam(feature_extractor_T.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.CrossEntropyLoss()

for param in feature_extractor_S.parameters():
    param.requires_grad = False


if torch.cuda.is_available():
    feature_extractor_T.cuda()
    feature_extractor_S.cuda()
    discriminator.cuda()
    loss.cuda()

feature_extractor_S.eval()
feature_extractor_T.train()
discriminator.train()

for epoch in range(epochs2):
    for i, (sdata, tdata) in enumerate(zip(dataloader_source, dataloader_target)):
        s_img, _ = sdata
        t_img, _ = tdata
        batch_size = min(s_img.shape[0],t_img.shape[0])

        s_img = s_img[0:batch_size, :, :, :]
        t_img = t_img[0:batch_size, :, :, :]

        if torch.cuda.is_available():
            s_img = Variable(s_img).cuda()
            t_img = Variable(t_img).cuda()

        # Train D
        optimizer_discriminator.zero_grad()

        featuresS_D = feature_extractor_S(s_img)
        featuresT_D = feature_extractor_T(t_img)

        predic_s_d = discriminator(featuresS_D)
        predic_t_d = discriminator(featuresT_D)
        predictionD = torch.cat((predic_s_d, predic_t_d), 0)


        label_s = Variable(torch.zeros(batch_size).long())
        label_t_d = Variable(torch.ones(batch_size).long())
        labels = torch.cat((label_s, label_t_d), 0).cuda()

        lossD = loss(predictionD, labels)
        lossD.backward()

        optimizer_discriminator.step()

        # Train target feature extractor
        optimizer_discriminator.zero_grad()
        optimizer_target.zero_grad()

        featuresT_T = feature_extractor_T(t_img)
        label_t = Variable(torch.zeros(batch_size).long()).cuda()

        predictionT = discriminator(featuresT_T)
        lossT = loss(predictionT, label_t)
        lossT.backward()

        optimizer_target.step()

        optimizer_target.zero_grad()



        if i % 10 == 0:
                print('[{}/{}]\tLoss disc: {:.6f}\tLoss disc(tgt): {:.6f}'.format(epoch, epochs2, lossD.item(), lossT.item()))

torch.save(feature_extractor_T.state_dict(), '{}target_encoder.pth'.format(model_root))
torch.save(discriminator.state_dict(), '{}discriminator.pth'.format(model_root))