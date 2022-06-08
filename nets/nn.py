from collections import defaultdict

import numpy
import torch
from mmcv.ops import nms
from mmcv.runner import BaseModule
from mmdet.core.bbox import transforms
from mmdet.models.builder import MODELS
from mmdet.models.detectors import TwoStageDetector
from mmdet.models.roi_heads import CascadeRoIHead
from torch.nn.functional import cross_entropy, one_hot, softmax


def build_detector(cfg, train_cfg=None, test_cfg=None):
    args = dict(train_cfg=train_cfg, test_cfg=test_cfg)
    return MODELS.build(cfg, default_args=args)


def mask2results(outputs, targets, num_classes):
    results = [[] for _ in range(num_classes)]
    for i in range(outputs.shape[0]):
        results[targets[i]].append(outputs[i])
    return results


@MODELS.register_module()
class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, epsilon=-1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets, *args, **kwargs):
        ce = cross_entropy(outputs, targets, reduction='none')
        pt = one_hot(targets, outputs.size()[1]) * softmax(outputs, dim=1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()


@MODELS.register_module()
class SparseMaskRCNN(TwoStageDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_rpn, 'SparseMaskRCNN do not support external proposals'

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert proposals is None, 'does not support external proposals'
        assert gt_masks is not None, 'needs mask ground-truth annotations for instance segmentation'

        x = self.extract_feat(img)
        proposal_boxes, proposal_features, images_wh_wh = self.rpn_head.forward_train(x, img_metas)
        roi_losses = self.roi_head.forward_train(x,
                                                 proposal_boxes,
                                                 proposal_features,
                                                 img_metas, gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore=gt_bboxes_ignore,
                                                 gt_masks=gt_masks, images_wh_wh=images_wh_wh)
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, images_wh_wh = self.rpn_head.simple_test_rpn(x, img_metas)
        results = self.roi_head.simple_test(x, proposal_boxes, proposal_features,
                                            img_metas, images_wh_wh=images_wh_wh, rescale=rescale)
        return results

    def aug_test(self, images, img_metas, rescale=False):
        x = self.extract_feats(images)
        proposal_boxes, proposal_features, images_wh_wh = self.rpn_head.aug_test_rpn(x, img_metas)
        results = self.roi_head.aug_test(x, proposal_boxes, proposal_features,
                                         img_metas, aug_images_wh_wh=images_wh_wh, rescale=rescale)
        return results

    def forward_dummy(self, img):
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_images = len(img)
        dummy_img_metas = [dict(img_shape=(800, 1333, 3)) for _ in range(num_images)]
        proposal_boxes, proposal_features, images_wh_wh = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes, proposal_features, dummy_img_metas)
        return roi_outs


@MODELS.register_module()
class SparseRPNHead(BaseModule):
    def __init__(self, num_proposals=100, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        self.fp16_enabled = False
        self.proposal_boxes = torch.nn.Embedding(num_proposals, 4)
        self.proposal_features = torch.nn.Embedding(num_proposals, 256)

    def init_weights(self):
        super().init_weights()
        torch.nn.init.constant_(self.proposal_boxes.weight[:, :2], 0.5)
        torch.nn.init.constant_(self.proposal_boxes.weight[:, 2:], 1)

    def decode_proposals(self, images, image_metas):
        proposals = self.proposal_boxes.weight.clone()
        proposals = transforms.bbox_cxcywh_to_xyxy(proposals)
        num_images = len(images[0])
        images_wh_wh = []
        for meta in image_metas:
            h, w, _ = meta['img_shape']
            images_wh_wh.append(images[0].new_tensor([[w, h, w, h]]))
        images_wh_wh = torch.cat(images_wh_wh, dim=0)
        images_wh_wh = images_wh_wh[:, None, :]

        proposals = proposals * images_wh_wh

        proposal_features = self.proposal_features.weight.clone()
        proposal_features = proposal_features[None].expand(num_images, *proposal_features.size())
        return proposals, proposal_features, images_wh_wh

    def forward_dummy(self, image, image_metas):
        return self.decode_proposals(image, image_metas)

    def forward_train(self, image, image_metas):
        return self.decode_proposals(image, image_metas)

    def simple_test_rpn(self, image, image_metas):
        return self.decode_proposals(image, image_metas)

    def aug_test_rpn(self, images, image_metas):
        aug_proposal_boxes = []
        aug_proposal_features = []
        aug_images_wh_wh = []
        for img, img_meta in zip(images, image_metas):
            proposal_boxes, proposal_features, images_wh_wh = self.simple_test_rpn(img, img_meta)
            aug_proposal_boxes.append(proposal_boxes)
            aug_proposal_features.append(proposal_features)
            aug_images_wh_wh.append(images_wh_wh)
        return aug_proposal_boxes, aug_proposal_features, aug_images_wh_wh


@MODELS.register_module()
class SparseMaskRoIHead(CascadeRoIHead):
    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=None,
                 mask_roi_extractor=None,
                 bbox_head=None, mask_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if bbox_roi_extractor is None:
            bbox_roi_extractor = dict(type='SingleRoIExtractor',
                                      roi_layer=dict(type='RoIAlign',
                                                     output_size=7, sampling_ratio=2),
                                      out_channels=256, featmap_strides=[4, 8, 16, 32])
        if mask_roi_extractor is None:
            mask_roi_extractor = dict(type='SingleRoIExtractor',
                                      roi_layer=dict(type='RoIAlign',
                                                     output_size=14, sampling_ratio=2),
                                      out_channels=256, featmap_strides=[4, 8, 16, 32])
        if bbox_head is None:
            bbox_head = dict(type='DIIHead',
                             num_classes=80,
                             num_ffn_fcs=2,
                             num_heads=8,
                             num_cls_fcs=1,
                             num_reg_fcs=3,
                             feedforward_channels=2048,
                             in_channels=256,
                             dropout=0.0,
                             ffn_act_cfg=dict(type='ReLU', inplace=True),
                             dynamic_conv_cfg=dict(type='DynamicConv',
                                                   in_channels=256,
                                                   feat_channels=64,
                                                   out_channels=256,
                                                   input_feat_shape=7,
                                                   act_cfg=dict(type='ReLU', inplace=True),
                                                   norm_cfg=dict(type='LN')),
                             loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                             loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                             loss_cls=dict(type='FocalLoss',
                                           use_sigmoid=True,
                                           gamma=2.0,
                                           alpha=0.25,
                                           loss_weight=2.0),
                             bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                             clip_border=False,
                                             target_means=[0., 0., 0., 0.],
                                             target_stds=[0.5, 0.5, 1., 1.]))
        if mask_head is None:
            mask_head = dict(type='DynamicMaskHead',
                             dynamic_conv_cfg=dict(type='DynamicConv',
                                                   in_channels=256,
                                                   feat_channels=64,
                                                   out_channels=256,
                                                   input_feat_shape=14,
                                                   with_proj=False,
                                                   act_cfg=dict(type='ReLU', inplace=True),
                                                   norm_cfg=dict(type='LN')),
                             num_convs=4,
                             num_classes=80,
                             roi_feat_size=14,
                             in_channels=256,
                             conv_kernel_size=3,
                             conv_out_channels=256,
                             class_agnostic=False,
                             norm_cfg=dict(type='BN'),
                             upsample_cfg=dict(type='deconv', scale_factor=2),
                             loss_mask=dict(type='DiceLoss',
                                            loss_weight=8.0,
                                            use_sigmoid=True, activate=False, eps=1e-5))
        assert bbox_roi_extractor is not None
        assert mask_roi_extractor is not None
        assert bbox_head is not None
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        super().__init__(num_stages, stage_loss_weights,
                         bbox_roi_extractor, bbox_head, mask_roi_extractor,
                         mask_head, None, train_cfg, test_cfg, pretrained, init_cfg)

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        num_images = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(bbox_feats, object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(rois,
                                                            rois.new_zeros(len(rois)),  # dummy arg
                                                            bbox_pred.view(-1, bbox_pred.size(-1)),
                                                            [rois.new_zeros(object_feats.size(1)) for _ in
                                                             range(num_images)], img_metas)
        bbox_results = dict(cls_score=cls_score,
                            decode_bbox_pred=torch.cat(proposal_list),
                            object_feats=object_feats,
                            attn_feats=attn_feats,
                            # detach then use it in label assign
                            detach_cls_score_list=[cls_score[i].detach() for i in range(num_images)],
                            detach_proposal_list=[item.detach() for item in proposal_list])

        return bbox_results

    def _mask_forward(self, stage, x, rois, attn_feats):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, attn_feats, sampling_results, gt_masks, train_cfg):
        if sum([len(gt_mask) for gt_mask in gt_masks]) == 0:
            print('Ground Truth Not Found!')
            loss_mask = sum([_.sum() for _ in self.mask_head[stage].parameters()]) * 0.
            return dict(loss_mask=loss_mask)
        pos_rois = transforms.bbox2roi([res.pos_bboxes for res in sampling_results])
        attn_feats = torch.cat([feats[res.pos_inds] for (feats, res) in zip(attn_feats, sampling_results)])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)

        mask_targets = self.mask_head[stage].get_targets(sampling_results, gt_masks, train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'], mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def forward_train(self, x,
                      proposal_boxes,
                      proposal_features,
                      img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, images_wh_wh=None, gt_masks=None):
        num_images = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        images_wh_wh = images_wh_wh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        losses = defaultdict(float)
        for stage in range(self.num_stages):
            rois = transforms.bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_images)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(num_images):
                norm_box_cc_wh = transforms.bbox_xyxy_to_cxcywh(proposal_list[i] / images_wh_wh[i])
                assign_result = self.bbox_assigner[stage].assign(norm_box_cc_wh, cls_pred_list[i],
                                                                 gt_bboxes[i], gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(assign_result,
                                                                  proposal_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(sampling_results,
                                                             gt_bboxes, gt_labels,
                                                             self.train_cfg[stage], True)
            cls_score = bbox_results['cls_score']
            object_feats = bbox_results['object_feats']
            decode_bbox_pred = bbox_results['decode_bbox_pred']

            loss = self.bbox_head[stage].loss(cls_score.view(-1, cls_score.size(-1)),
                                              decode_bbox_pred.view(-1, 4),
                                              *bbox_targets, imgs_whwh=images_wh_wh)

            if self.with_mask:
                mask_results = self._mask_forward_train(stage, x, bbox_results['attn_feats'],
                                                        sampling_results, gt_masks, self.train_cfg[stage])
                loss['loss_mask'] = mask_results['loss_mask']

            for key, value in loss.items():
                if 'loss' not in key:
                    continue
                if stage == 0:
                    losses[f'{key}'] = value * self.stage_loss_weights[stage]
                else:
                    losses[f'{key}'] += value * self.stage_loss_weights[stage]

        return losses

    def simple_test(self, x,
                    proposal_boxes,
                    proposal_features,
                    img_metas, images_wh_wh, rescale=False):
        assert self.with_bbox, 'Box head must be implemented.'
        # Decode initial proposals
        num_images = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_images)]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # "ms" in variable names means multi-stage
        ms_box_result = {}
        ms_seg_result = {}

        object_feats = proposal_features
        for stage in range(self.num_stages):
            rois = transforms.bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

        if self.with_mask:
            rois = transforms.bbox2roi(proposal_list)
            mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(num_images, -1,
                                                                          *mask_results['mask_pred'].size()[1:])

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_images):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(0, 1).topk(self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices // num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [transforms.bbox2result(det_bboxes[i], det_labels[i], num_classes)
                        for i in range(num_images)]
        ms_box_result['ensemble'] = bbox_results

        if self.with_mask:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                                 for scale_factor in scale_factors]
            _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                       for i in range(len(det_bboxes))]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_images):
                mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                segm_result = self.mask_head[-1].get_seg_masks(mask_pred_per_img,
                                                               _bboxes[img_id], det_labels[img_id],
                                                               self.test_cfg, ori_shapes[img_id],
                                                               scale_factors[img_id], rescale)
                segm_results.append(segm_result)

            ms_seg_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(zip(ms_box_result['ensemble'], ms_seg_result['ensemble']))
        else:
            results = ms_box_result['ensemble']

        return results

    def aug_test(self,
                 aug_x,
                 aug_proposal_boxes,
                 aug_proposal_features,
                 aug_img_metas,
                 aug_images_wh_wh,
                 rescale=False):

        samples_per_gpu = len(aug_img_metas[0])
        aug_det_bboxes = [[] for _ in range(samples_per_gpu)]
        aug_det_labels = [[] for _ in range(samples_per_gpu)]
        aug_mask_preds = [[] for _ in range(samples_per_gpu)]
        for x, proposal_boxes, proposal_features, img_metas, imgs_whwh in zip(aug_x,
                                                                              aug_proposal_boxes,
                                                                              aug_proposal_features,
                                                                              aug_img_metas, aug_images_wh_wh):

            num_images = len(img_metas)
            proposal_list = [proposal_boxes[i] for i in range(num_images)]
            ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            object_feats = proposal_features
            for stage in range(self.num_stages):
                rois = transforms.bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)
                object_feats = bbox_results['object_feats']
                cls_score = bbox_results['cls_score']
                proposal_list = bbox_results['detach_proposal_list']

            if self.with_mask:
                rois = transforms.bbox2roi(proposal_list)
                mask_results = self._mask_forward(stage, x, rois, bbox_results['attn_feats'])
                mask_results['mask_pred'] = mask_results['mask_pred'].reshape(num_images, -1,
                                                                              *mask_results['mask_pred'].size()[1:])

            num_classes = self.bbox_head[-1].num_classes
            det_bboxes = []
            det_labels = []

            if self.bbox_head[-1].loss_cls.use_sigmoid:
                cls_score = cls_score.sigmoid()
            else:
                cls_score = cls_score.softmax(-1)[..., :-1]

            for img_id in range(num_images):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(0, 1).topk(self.test_cfg.max_per_img,
                                                                                    sorted=False)
                labels_per_img = topk_indices % num_classes
                bbox_pred_per_img = proposal_list[img_id][topk_indices // num_classes]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                aug_det_bboxes[img_id].append(torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                det_bboxes.append(torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                aug_det_labels[img_id].append(labels_per_img)
                det_labels.append(labels_per_img)

            if self.with_mask:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                                     for scale_factor in scale_factors]
                _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                           for i in range(len(det_bboxes))]
                mask_pred = mask_results['mask_pred']
                for img_id in range(num_images):
                    mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices]
                    mask_pred_per_img = mask_pred_per_img[:, None, ...].repeat(1, num_classes, 1, 1)
                    segm_result = self.mask_head[-1].get_seg_masks(mask_pred_per_img,
                                                                   _bboxes[img_id], det_labels[img_id],
                                                                   self.test_cfg, ori_shapes[img_id],
                                                                   scale_factors[img_id], rescale, format=False)
                    aug_mask_preds[img_id].append(segm_result.detach().cpu().numpy())

        det_bboxes, det_labels, mask_preds = [], [], []

        for img_id in range(samples_per_gpu):
            for aug_id in range(len(aug_det_bboxes[img_id])):
                img_meta = aug_img_metas[aug_id][img_id]
                img_shape = img_meta['ori_shape']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                aug_det_bboxes[img_id][aug_id][:, :-1] = transforms.bbox_flip(aug_det_bboxes[img_id][aug_id][:, :-1],
                                                                              img_shape, flip_direction) if flip else \
                    aug_det_bboxes[img_id][aug_id][:, :-1]
                if flip:
                    if flip_direction == 'horizontal':
                        aug_mask_preds[img_id][aug_id] = aug_mask_preds[img_id][aug_id][:, :, ::-1]
                    else:
                        aug_mask_preds[img_id][aug_id] = aug_mask_preds[img_id][aug_id][:, ::-1, :]

        for img_id in range(samples_per_gpu):
            det_bboxes_per_im = torch.cat(aug_det_bboxes[img_id])
            det_labels_per_im = torch.cat(aug_det_labels[img_id])
            mask_preds_per_im = numpy.concatenate(aug_mask_preds[img_id])

            det_bboxes_per_im, keep_inds = nms.batched_nms(det_bboxes_per_im[:, :-1],
                                                           det_bboxes_per_im[:, -1].contiguous(),
                                                           det_labels_per_im, self.test_cfg.nms)
            det_bboxes_per_im = det_bboxes_per_im[:self.test_cfg.max_per_img, ...]
            det_labels_per_im = det_labels_per_im[keep_inds][:self.test_cfg.max_per_img, ...]
            mask_preds_per_im = mask_preds_per_im[keep_inds.detach().cpu().numpy()][:self.test_cfg.max_per_img, ...]
            det_bboxes.append(det_bboxes_per_im)
            det_labels.append(det_labels_per_im)
            mask_preds.append(mask_preds_per_im)

        ms_bbox_result = {}
        ms_segm_result = {}
        num_classes = self.bbox_head[-1].num_classes
        bbox_results = [transforms.bbox2result(det_bboxes[i], det_labels[i], num_classes)
                        for i in range(samples_per_gpu)]
        ms_bbox_result['ensemble'] = bbox_results
        mask_results = [mask2results(mask_preds[i], det_labels[i], num_classes)
                        for i in range(samples_per_gpu)]
        ms_segm_result['ensemble'] = mask_results

        if self.with_mask:
            results = list(zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']
        return results

    def forward_dummy(self, x, proposal_boxes, proposal_features, img_metas):
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = transforms.bbox2roi(proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
                all_stage_bbox_results.append(bbox_results)
                proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']
        return all_stage_bbox_results
