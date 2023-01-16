# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image

import torch
import torchvision.transforms as T
from models import build_model
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Graph
    parser.add_argument('--export_path', default="graph.json", type=str, help="Filename for the exported triples")

    parser.add_argument('--topk', default=10, type=int, help="Top k triples to return")

    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # VG classes
    CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
               'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
               'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
               'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
               'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
               'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
               'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
               'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
               'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
               'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
               'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
               'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
               'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
               'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                   'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                   'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                   'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                   'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                   'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # Extract relations, subjects and objects
    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    # get topk results. max(-1) means the last dimension, or in the 3d case, the "depth":
    # https://stackoverflow.com/questions/59702785/what-does-dim-1-or-2-mean-in-torch-sum
    # probas.shape == (200, len(REL_CLASSES)), probas_sub.shape == probas_obj.shape == (200, len(CLASSES))
    topk = args.topk
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    if (len(probas[keep_queries]) > 0):
        indices = torch.argsort(
            -probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
        keep_queries = keep_queries[indices]
    else:
        triples_to_json([], filename=args.export_path)
        print(f"RelTR found no matches for {args.img_path}")
        return

    # use lists to store the outputs via up-values
    # Currently we do not need attention regions (rendered as lights overlayed on the image)
    # conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    #
    # hooks = [
    #     model.backbone[-2].register_forward_hook(
    #         lambda self, input, output: conv_features.append(output)
    #     ),
    #     model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights_sub.append(output[1])
    #     ),
    #     model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights_obj.append(output[1])
    #     )
    # ]

    triples = []

    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        # These are also for attention
        # for hook in hooks:
        #     hook.remove()
        #
        # # don't need the list anymore
        # conv_features = conv_features[0]
        # dec_attn_weights_sub = dec_attn_weights_sub[0]
        # dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        # h, w = conv_features['0'].tensors.shape[-2:]
        # im_w, im_h = im.size

        for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            triple = {
                "subject": {
                    "id": CLASSES[probas_sub[idx].argmax()],
                    "xmin": sxmin.item(),
                    "ymin": symin.item(),
                    "xmax": sxmax.item(),
                    "ymax": symax.item()
                },
                "predicate": {
                    "id": REL_CLASSES[probas[idx].argmax()]
                },
                "object": {
                    "id": CLASSES[probas_obj[idx].argmax()],
                    "xmin": oxmin.item(),
                    "ymin": oymin.item(),
                    "xmax": oxmax.item(),
                    "ymax": oymax.item()
                }
            }
            triples.append(triple)
    triples_to_json(triples, filename=args.export_path)


def triples_to_json(triples, filename):
    with open(filename, "w") as file:
        file.write(json.dumps(triples))
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR triples', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
