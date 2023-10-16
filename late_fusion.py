import os
import cv2
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.ops import boxes as box_ops

CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

def avg_bbox_fusion(match_bbox_vec):
    avg_bboxs = np.sum(match_bbox_vec,axis=0) / len(match_bbox_vec)
    return avg_bboxs

def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox

def bayesian_fusion(match_score_vec):
    log_positive_scores = np.log(match_score_vec)
    log_negative_scores = np.log(1 - match_score_vec)
    fused_positive = np.exp(np.sum(log_positive_scores))
    fused_negative = np.exp(np.sum(log_negative_scores))
    fused_positive_normalized = fused_positive / (fused_positive + fused_negative)
    return fused_positive_normalized

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    # python3 late_fusion.py --img data --det1 yolov7/runs/detect/dip/labels --det2 FCOS/dip/FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x/labels --det3 FCOS/dip/FCOS_imprv_R_50_FPN_1x/labels --output results --method nms
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='img directory')
    parser.add_argument('--det1', help='#1 detection results')
    parser.add_argument('--det2', help='#2 detection results')
    parser.add_argument('--det3', default=None, help='#3 detection results')
    parser.add_argument('--output', default='results', help='path to output images directory')
    parser.add_argument('--method', default='nms', help='method to fuse boxes')
    args = parser.parse_args()

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in CATEGORIES]
    for path in tqdm(sorted(os.listdir(args.det1))):
        # det1
        with open(os.path.join(args.det1, path), 'r') as f:
            det1 = f.readlines()
            class1, box1, score1 = [], [], []
            for d in det1:
                d = d.split()
                class1.append(int(d[0]))
                box1.append(np.array([int(d[1]), int(d[2]), int(d[3]), int(d[4])]))
                score1.append(float(d[5]))
            
            class1 = np.array(class1)
            box1 = np.array(box1).reshape((-1, 4))
            score1 = np.array(score1)

        # det2
        with open(os.path.join(args.det2, path), 'r') as f:
            det2 = f.readlines()
            class2, box2, score2 = [], [], []
            for d in det2:
                d = d.split()
                class2.append(int(d[0]))
                box2.append(np.array([int(d[1]), int(d[2]), int(d[3]), int(d[4])]))
                score2.append(float(d[5]))
            
            class2 = np.array(class2)
            box2 = np.array(box2).reshape((-1, 4))
            score2 = np.array(score2)
        
        # det3
        if args.det3 != None:
            with open(os.path.join(args.det3, path), 'r') as f:
                det3 = f.readlines()
                class3, box3, score3 = [], [], []
                for d in det3:
                    d = d.split()
                    class3.append(int(d[0]))
                    box3.append(np.array([int(d[1]), int(d[2]), int(d[3]), int(d[4])]))
                    score3.append(float(d[5]))
                
                class3 = np.array(class3)
                box3 = np.array(box3).reshape((-1, 4))
                score3 = np.array(score3)

        # prepare data
        image = cv2.imread(os.path.join(args.img, path.replace('txt', 'png')))
        if args.det3 != None:
            classes = np.concatenate((class1, class2, class3), axis=0)
            boxes = np.concatenate((box1, box2, box3), axis=0)
            scores = np.concatenate((score1, score2, score3), axis=0)
        else:
            classes = np.concatenate((class1, class2), axis=0)
            boxes = np.concatenate((box1, box2), axis=0)
            scores = np.concatenate((score1, score2), axis=0)

        if args.method == 'nms':
            classes = torch.Tensor(classes)
            boxes = torch.Tensor(boxes)
            scores = torch.Tensor(scores)

            # keep_id = box_ops.batched_nms(boxes, scores, classes, iou_threshold=0.5)
            keep_id = box_ops.nms(boxes, scores, iou_threshold=0.5)
            classes = classes[keep_id]
            boxes = boxes[keep_id]
            scores = scores[keep_id]

            for i, box in enumerate(boxes):
                label = f'{CATEGORIES[int(classes[i])]} {scores[i]:.2f}'
                plot_one_box(box, image, color=colors[int(classes[i])], label=label, line_thickness=1)
            cv2.imwrite(os.path.join(args.output, 'nms', path.replace('txt', 'png')), image)

        elif args.method == 'bayesian':
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            match_scores = []
            match_bboxs = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                
                inds = np.where(ovr <= 0.5)[0]
                match = np.where(ovr > 0.5)[0]
                match_ind = order[match + 1]
                
                match_score = list(scores[match_ind])
                match_bbox = list(boxes[match_ind][:, :4])
                original_score = scores[i].tolist()
                original_bbox = boxes[i][:4]

                if len(match_score) > 0:
                    match_score += [original_score]
                    match_bbox += [original_bbox]
                    final_score = bayesian_fusion(np.array(match_score))
                    final_bbox = avg_bbox_fusion(match_bbox)
                    match_scores.append(final_score)
                    match_bboxs.append(final_bbox)
                else:
                    match_scores.append(original_score)
                    match_bboxs.append(original_bbox)

                order = order[inds + 1]

            assert len(keep) == len(match_scores)
            assert len(keep) == len(match_bboxs)
            match_classes = classes[keep]

            for i, box in enumerate(match_bboxs):
                label = f'{CATEGORIES[int(match_classes[i])]} {match_scores[i]:.2f}'
                plot_one_box(box, image, color=colors[int(match_classes[i])], label=label, line_thickness=1)
            cv2.imwrite(os.path.join(args.output, 'bayesian', path.replace('txt', 'png')), image)