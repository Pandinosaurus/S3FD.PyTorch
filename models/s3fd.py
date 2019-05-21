import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.modules.l2norm import L2Norm
from models.mobilenet_v2_xiaomi import MobileNetV2, InvertedResidual
import numpy as np
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

class S3FD(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(S3FD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512]
        self.vgg = nn.ModuleList(vgg(vgg_cfg, 3))

        extras_cfg = [256, 'S', 512, 128, 'S', 256]
        self.extras = nn.ModuleList(add_extras(extras_cfg, 1024))

        self.conv3_3_L2Norm = L2Norm(256, 10)
        self.conv4_3_L2Norm = L2Norm(512, 8)
        self.conv5_3_L2Norm = L2Norm(512, 5)
        
        self.loc, self.conf = self.multibox(self.num_classes)
        
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

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
        # Max-out BG label
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        #conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(512, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(512, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(1024, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(1024, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(512, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
    
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        detection_dimension = list()

        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 15 == k:
                s = self.conv3_3_L2Norm(x)
                sources.append(s)
                detection_dimension.append(x.shape[2:])
            elif 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
                detection_dimension.append(x.shape[2:])
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)
                detection_dimension.append(x.shape[2:])

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        detection_dimension.append(x.shape[2:])

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                detection_dimension.append(x.shape[2:])
        
        detection_dimension = torch.Tensor(detection_dimension)
        detection_dimension = detection_dimension.cuda()

        for index, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
            #print(x.size())
            if index != 0:
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            else:
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf_t = c(x)
                max_conf, _ = conf_t[:, 0:3, :, :].max(1, keepdim=True)
                lab_conf = conf_t[:, 3:, :, :]
                out_conf = torch.cat((max_conf, lab_conf), dim=1)
                conf.append(out_conf.permute(0, 2, 3, 1).contiguous())

        '''for index, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())'''
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (loc.view(loc.size(0), -1, 4),
                        self.softmax(conf.view(-1, self.num_classes)),
                        detection_dimension)
        else:
            output = (loc.view(loc.size(0), -1, 4),
                        conf.view(conf.size(0), -1, self.num_classes),
                        detection_dimension)
    
        return output


class S3FD_MV2(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(S3FD_MV2, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.detection_dimension = list()
        self.base_net = MobileNetV2(width_mult=1.0).features
        self.source_layer_indexes = [
            GraphPath(7, 'conv', 3),
            GraphPath(14, 'conv', 3),
            19,
        ]
        print(self.base_net)

        self.extras = nn.ModuleList([
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5)
        ])

        self.conv3_3_L2Norm = L2Norm(192, 10)
        self.conv4_3_L2Norm = L2Norm(576, 8)
        self.conv5_3_L2Norm = L2Norm(1280, 5)

        self.loc, self.conf = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

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
        # Max-out BG label
        loc_layers += [nn.Conv2d(192, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(192, 1 * 4, kernel_size=3, padding=1)]
        # conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        # conv4_3
        loc_layers += [nn.Conv2d(576, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(576, 1 * num_classes, kernel_size=3, padding=1)]
        # conv5_3
        loc_layers += [nn.Conv2d(1280, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(1280, 1 * num_classes, kernel_size=3, padding=1)]
        # fc6
        loc_layers += [nn.Conv2d(512, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, 1 * num_classes, kernel_size=3, padding=1)]
        # fc7
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        # conv7_2
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def forward(self, x: torch.Tensor):
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        self.detection_dimension = torch.Tensor(self.detection_dimension)

        if self.phase == "test":
            output = (locations,
                      self.softmax(confidences),
                      self.detection_dimension)
        else:
            output = (locations,
                      confidences,
                      self.detection_dimension)

        return output

    def compute_header(self, i, x):

        # add detection feature map
        # print(x.shape[2:])
        self.detection_dimension.append(x.shape[2:])
        # add extra normalization
        if i == 0:
            x =  self.conv3_3_L2Norm(x)
        elif i == 1:
            x = self.conv4_3_L2Norm(x)
        elif i == 2:
            x = self.conv5_3_L2Norm(x)

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

        return confidence, location

if __name__ == '__main__':
    net = S3FD_MV2('train', 640, 2)
    image = np.zeros((1,3,640,640), dtype=np.float32)
    output = net.forward(torch.from_numpy(image))
    print(output)
