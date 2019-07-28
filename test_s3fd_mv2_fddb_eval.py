from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data.config_s3fd_mv2 import cfg
from layers.functions.prior_box_s3fd import PriorBox
from utils.nms_wrapper import nms
import cv2
from models.s3fd import S3FD_MV2
from utils.box_utils import decode
from utils.timer import Timer
import scipy.io as sio


def parse_args(argv):
    parser = argparse.ArgumentParser(description='S3FD')
    parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
    parser.add_argument('--confidence_threshold', default=0.1, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--image_dir', type=str, help='images directory')
    parser.add_argument('--file_list', type=str, help='images list')
    parser.add_argument('--file_result', type=str, help='result of detection')
    args = parser.parse_args(argv)
    return args


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect_face(net, img, resize):
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    if args.cuda:
        img = img.cuda()
        scale = scale.cuda()

    out = net(img)  # forward pass
    priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()
    loc, conf, _ = out
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.data.squeeze(0).cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores >= args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]
    #print(boxes)

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, args.nms_threshold)
    dets = dets[keep, :]
    #print(dets)

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    return dets

def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t

def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    # net and model
    net = S3FD_MV2(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model)
    net.eval()
    print('Finished loading model!')
    print(net)
    print(args.cuda)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    image_dir = args.image_dir
    file_list = args.file_list
    file_result = args.file_result

    if not os.path.exists(image_dir):
        print("image_dir: {} does not exist".format(image_dir))
        exit()
    if not os.path.exists(file_list):
        print("file_list: {} does not exist".format(file_list))
        exit()

    outputname = os.path.basename(image_dir[:-1]) if image_dir[-1] == '/' else os.path.basename(image_dir)
    print(outputname)
    output_dir = outputname + "_" + os.path.basename(args.trained_model).split(".")[0].lower() + "_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    imgs_path_fd = open(file_list, "r")
    imgs_path = imgs_path_fd.readlines()
    imgs_path_fd.close()

    count = 0
    _str = ""

    for img_path in imgs_path:
        full_path = os.path.join(image_dir, img_path.strip("\n") + ".jpg")
        print(full_path)
        img = cv2.imread(full_path)
        img_float = np.float32(img)

        h,w,c = np.shape(img_float)
        minside = h if h < w else w

        # resize = 300.0 / minside
        resize = 1.0

        dets = detect_face(net, img_float, resize)  # origin test


        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        s  = dets[:, 4]

        for i in range(dets.shape[0]):
            if s[i] >= 0.9:
                cv2.rectangle(img, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, img_path.replace("/","_").strip("\n") + ".jpg"), img)

        str_name = img_path.strip("\n") + "\n"
        str_box = ""
        count = 0
        for i in range(dets.shape[0]):
            str_box += str(x1[i]) + " " \
                       + str(y1[i]) + " " \
                       + str(x2[i] - x1[i]) + " " \
                       + str(y2[i] - y1[i]) + " " \
                       + str(s[i]) + "\n"
            count += 1

        _str += str_name
        _str += str(count) + "\n"
        _str += str_box

        print(str_name)
        print(count)
        print(str_box)

    d_ret_fd = open(file_result, "w")
    d_ret_fd.writelines(_str)
    d_ret_fd.close()