import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall
from .mean_ap import eval_map
#from mmdet import datasets

class EvalHook(Hook):
    """Evaluation hook.
    Notes:
        If new arguments are added for EvalHook, tools/test.py may be
    effected.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, dataloader, start=None, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        if not interval > 0:
            raise ValueError(f'interval must be positive, but got {interval}')
        if start is not None and start < 0:
            warnings.warn(
                f'The evaluation start epoch {start} is smaller than 0, '
                f'use 0 instead', UserWarning)
            start = 0
        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.eval_kwargs = eval_kwargs
        self.initial_epoch_flag = True

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training."""
        if not self.initial_epoch_flag:
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_epoch_flag = False

    def evaluation_flag(self, runner):
        """Judge whether to perform_evaluation after this epoch.
        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                # No evaluation during the interval epochs.
                return False
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False
        return True

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        

class DistEvalHook(EvalHook):
    """Distributed evaluation hook.
    Notes:
        If new arguments are added, tools/test.py may be effected.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 tmpdir=None,
                 gpu_collect=False,
                 **eval_kwargs):
        super().__init__(
            dataloader, start=start, interval=interval, **eval_kwargs)
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return
        from mmdet.apis import multi_gpu_test
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

#class DistEvalHook(Hook):
#
#    def __init__(self, dataset, interval=1):
#        if isinstance(dataset, Dataset):
#            self.dataset = dataset
#        elif isinstance(dataset, dict):
#            self.dataset = obj_from_dict(dataset, datasets,
#                                         {'test_mode': True})
#        else:
#            raise TypeError(
#                'dataset must be a Dataset object or a dict, not {}'.format(
#                    type(dataset)))
#        self.interval = interval
#
#    def after_train_epoch(self, runner):
#        if not self.every_n_epochs(runner, self.interval):
#            return
#        runner.model.eval()
#        results = [None for _ in range(len(self.dataset))]
#        if runner.rank == 0:
#            prog_bar = mmcv.ProgressBar(len(self.dataset))
#        for idx in range(runner.rank, len(self.dataset), runner.world_size):
#            data = self.dataset[idx]
#            data_gpu = scatter(
#                collate([data], samples_per_gpu=1),
#                [torch.cuda.current_device()])[0]
#
#            # compute output
#            with torch.no_grad():
#                result = runner.model(
#                    return_loss=False, rescale=True, **data_gpu)
#            results[idx] = result
#
#            batch_size = runner.world_size
#            if runner.rank == 0:
#                for _ in range(batch_size):
#                    prog_bar.update()
#
#        if runner.rank == 0:
#            print('\n')
#            dist.barrier()
#            for i in range(1, runner.world_size):
#                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
#                tmp_results = mmcv.load(tmp_file)
#                for idx in range(i, len(results), runner.world_size):
#                    results[idx] = tmp_results[idx]
#                os.remove(tmp_file)
#            self.evaluate(runner, results)
#        else:
#            tmp_file = osp.join(runner.work_dir,
#                                'temp_{}.pkl'.format(runner.rank))
#            mmcv.dump(results, tmp_file)
#            dist.barrier()
#        dist.barrier()
#
#    def evaluate(self):
#        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        os.remove(tmp_file)
