import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock
from layers.modules.l2norm import L2Norm


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_uniform(m.weight.data)

class S3FD_RESNET18(nn.Module):

    def __init__(self, phase, size, num_classes):
        super().__init__()
        self.fo = 256
        self.fi = 512
        self.arch = 'resnet18'
        self.phase = phase
        self.num_classes = num_classes
        self.base = getattr(models, self.arch)(pretrained=False)

        self.featuremap = nn.ModuleList([
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.base._make_layer(BasicBlock, self.fo, blocks=1, stride=2),
            self.base._make_layer(BasicBlock, self.fo, blocks=1, stride=2),
            self.base._make_layer(BasicBlock, 64, blocks=1, stride=2)
        ])

        self.conv2_2_L2Norm = L2Norm(128, 10)
        self.conv3_2_L2Norm = L2Norm(256, 8)
        self.conv4_2_L2Norm = L2Norm(512, 5)

        self.loc, self.conf = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        # we load pretrained model from outside
        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        f_in = [128, 256, 512, self.fo, self.fo, 64]

        # Max-out BG label
        loc_layers += [nn.Conv2d(f_in[0], 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(f_in[0], 1 * 4, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(f_in[1], 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(f_in[1], 1 * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(f_in[2], 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(f_in[2], 1 * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(f_in[3], 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(f_in[3], 1 * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(f_in[4], 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(f_in[4], 1 * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(f_in[5], 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(f_in[5], 1 * num_classes, kernel_size=3, padding=1)]

        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def compute_header(self, i, x):
        # add extra normalization
        if i == 0:
            x =  self.conv2_2_L2Norm(x)
        elif i == 1:
            x = self.conv3_2_L2Norm(x)
        elif i == 2:
            x = self.conv4_2_L2Norm(x)

        # maxout
        if i == 0:
            conf_t = self.conf[i](x)
            max_conf, _ = conf_t[:, 0:3, :, :].max(1, keepdim=True)
            lab_conf = conf_t[:, 3:, :, :]
            confidence = torch.cat((max_conf, lab_conf), dim=1)
        else:
            confidence = self.conf[i](x)

        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.loc[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location, x.shape[2:]

    def forward(self, x):
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        detection_dimension = list()

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)

        for idx, layer in enumerate(self.featuremap):
            x = layer(x)
            confidence, location, dims = self.compute_header(idx, x)
            confidences.append(confidence)
            locations.append(location)
            detection_dimension.append(dims)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        detection_dimension = torch.Tensor(detection_dimension)

        if self.phase == "test":
            output = (locations,
                      self.softmax(confidences),
                      detection_dimension)
        else:
            output = (locations,
                      confidences,
                      detection_dimension)

        return output


if __name__ == '__main__':
    import numpy as np
    net = S3FD_RESNET18('train', 640, 2)
    print(net)
    image = np.zeros((1,3,640,640), dtype=np.float32)
    output = net.forward(torch.from_numpy(image))
    print(output)
