import torch.nn as nn
import torch
import torch.nn.init as init

'''
Grad reverse taken from: https://github.com/CuthbertCai/pytorch_DANN/blob/master/models/models.py
'''

class feature_extractor(nn.Module):

    def __init__(self):

        super(feature_extractor,self).__init__()

        self.main = nn.Sequential(
            nn.Sequential(

                nn.Conv2d(3, 64, 5),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(True),

                nn.Conv2d(64, 50, 5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(False),
                nn.MaxPool2d(2),
                nn.ReLU(True)
            )
        )

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        x = self.main(x)
        x = x.view(-1, 50 * 4 * 4)

        return x



class class_classifier(nn.Module):
    def __init__(self):
        super(class_classifier, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(50*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(False),

            nn.Linear(100,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100,10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.main(x)

        return x


class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(50*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100,2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, constant):

        x = GradReverse.grad_reverse(x, constant)
        x = self.main(x)

        return x

class svhn_feature_extractor(nn.Module):

    def __init__(self):

        super(svhn_feature_extractor,self).__init__()

        self.main = nn.Sequential(
            nn.Sequential(

                nn.Conv2d(3, 64, 5),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.ReLU(True),

                nn.Conv2d(64, 64, 5, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.Conv2d(64,128,5,stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Dropout2d(False),

            )
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        x = self.main(x)
        x = x.view(-1, 128 * 3 * 3)

        return x

class svhn_class_classifier(nn.Module):
    def __init__(self):
        super(svhn_class_classifier, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(128*3*3, 3072),
            nn.BatchNorm1d(3072),
            nn.ReLU(True),
            nn.Dropout(False),

            nn.Linear(3072,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048,10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.main(x)

        return x


class svhn_domain_classifier(nn.Module):
    def __init__(self):
        super(svhn_domain_classifier, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(128*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(False),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(False),
            nn.Linear(1024,2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, constant):

        x = GradReverse.grad_reverse(x, constant)
        x = self.main(x)

        return x





class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataset import DANN
    img_size = 28
    name = 'mnistm'

    train_dataset = DANN(name=name, img_size=img_size, train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

    l = iter(train_loader)

    img,t = next(l)

    g = svhn_feature_extractor()
    h = g(img)