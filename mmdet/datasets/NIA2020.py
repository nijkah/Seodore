from .coco import CocoDataset
import numpy as np
from .builder import DATASETS

@DATASETS.register_module()
class NIA2020(CocoDataset):

    CLASSES= ('small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car', 'bus', 'truck', 'train',
            'crane', 'bridge', 'oil tank', 'dam', 'athletic field', 'helipad', 'roundabout')

@DATASETS.register_module()
class NIA2020_20(CocoDataset):

    CLASSES= ('background', 'small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car', 'bus', 'truck', 'train',
            'crane', 'bridge', 'oil tank', 'dam', 'indoor playground', 'outdoor_playground', 'helipad', 'roundabout',
            'helicopter', 'individual container', 'grouped container', 'swimming pool')

