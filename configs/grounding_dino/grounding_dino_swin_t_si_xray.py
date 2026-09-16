import os

_base_ = 'grounding_dino_swin-t_finetune_8xb2_20e_cat.py'

project_root = os.environ.get(
    'SCI_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
data_root = project_root + '/'
class_name = ('gun', 'knife', 'wrench', 'plier', 'scissor')
num_classes = len(class_name)
lang_model_name = 'bert-base-uncased'

metainfo = dict(
    classes=class_name,
    palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
             (255, 0, 255)])

model = dict(
    language_model=dict(name=lang_model_name),
    bbox_head=dict(num_classes=num_classes))
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/si_xray_d/train.json',
        data_prefix=dict(img='dataset/SIXray-D/images/train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1)))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/si_xray_d/val.json',
        data_prefix=dict(img='dataset/SIXray-D/images/val/'),
        metainfo=metainfo,
        test_mode=True))

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/si_xray_d/test.json',
        data_prefix=dict(img='dataset/SIXray-D/images/test/'),
        metainfo=metainfo,
        test_mode=True))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'annotations/si_xray_d/val.json'),
    metric='bbox',
    classwise=True)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, 'annotations/si_xray_d/test.json'),
    metric='bbox',
    classwise=True)

max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[16, 19], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='coco/bbox_mAP', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))
load_from = os.path.join(
    project_root,
    'mmdetection/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth')
