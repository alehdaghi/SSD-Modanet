# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np

from apex.fp16_utils import network_to_half

from dle.inference import prepare_input
from ssd.model import SSD300, ResNet, MobileNet
from ssd.utils import dboxes300_coco, Encoder, draw_patches, modanet_categories


def load_checkpoint(model, model_file):
    cp = torch.load(model_file)['model']
    model.load_state_dict(cp)


def build_predictor(model_file, backbone='resnet50'):
    if backbone.startswith("resnet") :
        ssd300 = SSD300(backbone=ResNet(backbone))
    else:
        ssd300 = SSD300(backbone=MobileNet(backbone))
    load_checkpoint(ssd300, model_file)

    return ssd300


def prepare_model(checkpoint_path, backbone='resnet50'):
    ssd300 = build_predictor(checkpoint_path, backbone)
    ssd300 = ssd300.cuda()
    ssd300 = network_to_half(ssd300)
    ssd300 = ssd300.eval()

    return ssd300

import torchvision.transforms as transforms
def prepare_tensor(inputs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    NHWC = np.array(inputs)
    NCHW = np.swapaxes(np.swapaxes(NHWC, 2, 3), 1, 2)
    tensor = torch.from_numpy(NCHW)
    tensor = normalize(tensor)
    tensor = tensor.cuda()
    tensor = tensor.half()

    return tensor


def decode_results(predictions):
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ploc, plabel = [val.float() for val in predictions]
    results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)

    return [ [ pred.detach().cpu().numpy()   for pred in detections ]
             for detections in results
           ]


def pick_best(detections, treshold):
    bboxes, classes, confidences = detections
    best = np.argwhere(confidences > treshold).squeeze(axis=1)

    return [pred[best] for pred in detections]


def main(checkpoint_path, imgs, backbone='resnet18'):
    inputs = [prepare_input(uri) for uri in imgs]
    tensor = prepare_tensor(inputs)
    ssd300 = prepare_model(checkpoint_path, backbone=backbone)

    predictions = ssd300(tensor)

    results = decode_results(predictions)
    best_results = [pick_best(detections, treshold=0.3) for detections in results]
    return best_results, inputs

if __name__ == '__main__':
    best_results, imgs = main(
            checkpoint_path='../models/resnet50_299.pt',
            imgs=['/home/mahdi/PycharmProjects/Datasets/SYSU-MM01/cam1/0008/0012.jpg',
                '/media/mahdi/2e197b57-e3e6-4185-8d1b-5fbb1c3b8b55/datasets/modanet/images/0000003.jpg',
                 ],
            backbone='resnet50'
    )
    label_map = modanet_categories()
    draw_patches(img=imgs[0], bboxes=best_results[0][0], labels=best_results[0][1], label_map=label_map)
    print(best_results)
