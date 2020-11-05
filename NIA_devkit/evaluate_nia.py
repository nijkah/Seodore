# This code is modified to understand mAP with rotated bounding box from https://github.com/facebookresearch/maskrcnn-benchmark
# --- Original comments --
# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)

from __future__ import division

import os
from collections import defaultdict
import argparse
import itertools
#from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

def compute_iou(rbox_info):
    rbox1, rbox2 = rbox_info
    poly1 = Polygon([[rbox1[idx], rbox1[idx + 1]] for idx in range(0, 8, 2)])
    poly2 = Polygon([[rbox2[idx], rbox2[idx + 1]] for idx in range(0, 8, 2)])

    try:
        inter = poly1.intersection(poly2).area
    except:
        inter = 0

    return inter / (poly1.area + poly2.area - inter)

def rboxlist_iou_fast(rboxlist1, rboxlist2):
    """ compute IoU between rotated bounding boxes using Shapely.Polygon

    Computes pairwise intersection-over-union between rbox collections.

    Args:
      rboxlist1: N rboxes.
      rboxlist2: M rboxes.

    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """

    #iou = np.zeros((len(rboxlist1), len(rboxlist2)))
    # ((idx1, rbox1), (idx2, rbox2))
    #paramlist = itertools.product(enumerate(rboxlist1), enumerate(rboxlist2))
    paramlist = itertools.product(rboxlist1, rboxlist2)
    N, M = len(rboxlist1), len(rboxlist2)
    with mp.Pool(32) as pool:
        results = pool.map(compute_iou, paramlist)
    #with ProcessPoolExecutor(32) as executor:
    #    results = executor.map(compute_iou, paramlist)
    iou = np.array(results).reshape(N, M)


    #for idx1, rbox1 in enumerate(rboxlist1):
    #    poly1 = Polygon([[rbox1[idx], rbox1[idx + 1]] for idx in range(0, 8, 2)])
    #    for idx2, rbox2 in enumerate(rboxlist2):
    #        poly2 = Polygon([[rbox2[idx], rbox2[idx + 1]] for idx in range(0, 8, 2)])
    #        try:
    #            inter = poly1.intersection(poly2).area
    #        except:
    #            inter = 0
    #        iou[idx1, idx2] = inter / (poly1.area + poly2.area - inter)

    return iou

def rboxlist_iou(rboxlist1, rboxlist2):
    """ compute IoU between rotated bounding boxes using Shapely.Polygon

    Computes pairwise intersection-over-union between rbox collections.

    Args:
      rboxlist1: N rboxes.
      rboxlist2: M rboxes.

    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """

    iou = np.zeros((len(rboxlist1), len(rboxlist2)))
    for idx1, rbox1 in enumerate(rboxlist1):
        poly1 = Polygon([[rbox1[idx], rbox1[idx + 1]] for idx in range(0, 8, 2)])
        for idx2, rbox2 in enumerate(rboxlist2):
            poly2 = Polygon([[rbox2[idx], rbox2[idx + 1]] for idx in range(0, 8, 2)])
            try:
                inter = poly1.intersection(poly2).area
            except:
                inter = 0
            iou[idx1, idx2] = inter / (poly1.area + poly2.area - inter)

    return iou


def do_nia_evaluation(gt_csv_path, pred_csv_path, output_folder=None):
    gts_df = pd.read_csv(gt_csv_path)
    preds_df = pd.read_csv(pred_csv_path)

    pred_boxlists = []
    gt_boxlists = []
    for _, image_id in enumerate(gts_df.file_name.unique()):
        pred_df_by_image_id = preds_df[preds_df.file_name == image_id]
        pred_boxlists.append(pred_df_by_image_id)

        gt_df_by_image_id = gts_df[gts_df.file_name == image_id]
        gt_boxlists.append(gt_df_by_image_id)

    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=False,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(i, ap)

    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = np.array(
            [(rbox.point1_x, rbox.point1_y,
              rbox.point2_x, rbox.point2_y,
              rbox.point3_x, rbox.point3_y,
              rbox.point4_x, rbox.point4_y) for _, rbox in pred_boxlist.iterrows()])
        pred_label = np.array([rbox.class_id for _, rbox in pred_boxlist.iterrows()])
        pred_score = np.array([rbox.confidence for _, rbox in pred_boxlist.iterrows()])

        gt_bbox = np.array(
            [(rbox.point1_x, rbox.point1_y,
              rbox.point2_x, rbox.point2_y,
              rbox.point3_x, rbox.point3_y,
              rbox.point4_x, rbox.point4_y) for _, rbox in gt_boxlist.iterrows()])
        gt_label = np.array([rbox.class_id for _, rbox in gt_boxlist.iterrows()])

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]

            n_pos[l] += len(gt_bbox_l)
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = rboxlist_iou_fast(
                pred_bbox_l, gt_bbox_l
            )
            #iou = rboxlist_iou(
            #    pred_bbox_l, gt_bbox_l
            #)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_csv_path', type=str)
    parser.add_argument('--pred_csv_path', type=str)
    args = parser.parse_args()

    map = do_nia_evaluation(args.gt_csv_path, args.pred_csv_path)
    print('mAP:', map)
