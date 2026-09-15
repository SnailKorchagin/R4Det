# Copyright (c) Facebook, Inc. and its affiliates.
import importlib.util

from detectron2.layers import ShapeSpec

from .backbone import (
    BACKBONE_REGISTRY,
    FPN,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_backbone,
    build_resnet_backbone,
    make_stage,
)

# The R4Det release vendors a trimmed Detectron2 tree without detectron2.data.
# Meta-architectures and TTA depend on that optional package, while the backbone,
# layers, structures, and ROI pooler used by R4Det do not.
if (
    importlib.util.find_spec("detectron2.data") is not None
    and importlib.util.find_spec("detectron2.utils.events") is not None
):
    from .anchor_generator import build_anchor_generator, ANCHOR_GENERATOR_REGISTRY
    from .meta_arch import (
        META_ARCH_REGISTRY,
        SEM_SEG_HEADS_REGISTRY,
        GeneralizedRCNN,
        PanopticFPN,
        ProposalNetwork,
        RetinaNet,
        SemanticSegmentor,
        build_model,
        build_sem_seg_head,
        FCOS,
    )
    from .postprocessing import detector_postprocess
    from .proposal_generator import (
        PROPOSAL_GENERATOR_REGISTRY,
        build_proposal_generator,
        RPN_HEAD_REGISTRY,
        build_rpn_head,
    )
    from .roi_heads import (
        ROI_BOX_HEAD_REGISTRY,
        ROI_HEADS_REGISTRY,
        ROI_KEYPOINT_HEAD_REGISTRY,
        ROI_MASK_HEAD_REGISTRY,
        ROIHeads,
        StandardROIHeads,
        BaseMaskRCNNHead,
        BaseKeypointRCNNHead,
        FastRCNNOutputLayers,
        build_box_head,
        build_keypoint_head,
        build_mask_head,
        build_roi_heads,
    )
    from .test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA
    from .mmdet_wrapper import MMDetBackbone, MMDetDetector

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
