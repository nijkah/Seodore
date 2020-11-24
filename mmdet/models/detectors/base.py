import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn
import pycocotools.mask as maskUtils

from mmdet.core import tensor2imgs, get_classes


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
    #def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
    #def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            # print('in base detector')
            # import pdb
            # pdb.set_trace()
            #return self.forward_train(img, img_meta, **kwargs)
            return self.forward_train(img, img_metas, **kwargs)
        else:
            #return self.forward_test(img, img_meta, **kwargs)
            return self.forward_test(img, img_metas, **kwargs)

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img


    #def show_result(self,
    #                data,
    #                result,
    #                img_norm_cfg,
    #                dataset=None,
    #                score_thr=0.3):
    #    if isinstance(result, tuple):
    #        bbox_result, segm_result = result
    #    else:
    #        bbox_result, segm_result = result, None

    #    img_tensor = data['img'][0]
    #    img_metas = data['img_meta'][0].data[0]
    #    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    #    assert len(imgs) == len(img_metas)

    #    if dataset is None:
    #        class_names = self.CLASSES
    #    elif isinstance(dataset, str):
    #        class_names = get_classes(dataset)
    #    elif isinstance(dataset, (list, tuple)):
    #        class_names = dataset
    #    else:
    #        raise TypeError(
    #            'dataset must be a valid dataset name or a sequence'
    #            ' of class names, not {}'.format(type(dataset)))

    #    for img, img_meta in zip(imgs, img_metas):
    #        h, w, _ = img_meta['img_shape']
    #        img_show = img[:h, :w, :]

    #        bboxes = np.vstack(bbox_result)
    #        # draw segmentation masks
    #        if segm_result is not None:
    #            segms = mmcv.concat_list(segm_result)
    #            inds = np.where(bboxes[:, -1] > score_thr)[0]
    #            for i in inds:
    #                color_mask = np.random.randint(
    #                    0, 256, (1, 3), dtype=np.uint8)
    #                mask = maskUtils.decode(segms[i]).astype(np.bool)
    #                img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
    #        # draw bounding boxes
    #        labels = [
    #            np.full(bbox.shape[0], i, dtype=np.int32)
    #            for i, bbox in enumerate(bbox_result)
    #        ]
    #        labels = np.concatenate(labels)
    #        mmcv.imshow_det_bboxes(
    #            img_show,
    #            bboxes,
    #            labels,
    #            class_names=class_names,
    #            score_thr=score_thr)
