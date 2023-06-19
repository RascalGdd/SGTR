import os
import os.path as osp

import cvpods
# this is a linked file on server
# config.py -> config_oiv6.py
from cvpods.configs.rel_detr_config import OneStageRelDetrBASEConfig

cvpods_home = osp.dirname(cvpods.__path__[0])
curr_folder = '/'.join(osp.realpath(__file__).split('/')[:-1])

aux_loss_weight = 0.5

rel_dec_layer = 6
ent_dec_layer = 6

_config_dict = dict(
    DEBUG=False,
    DUMP_INTERMEDITE=False,

    EXPERIMENT_NAME=f"4dor-SGTR-rel_dec-{rel_dec_layer}",
    # the program will use this config to uptate the initial config(over write the existed, ignore doesnt existed)
    OVERIDE_CFG_DIR="",


    MODEL=dict(
        WEIGHTS_LOAD_MAPPING={
            "obj_class_embed": "class_embed",
            "obj_bbox_embed": "bbox_embed",
            "sub_class_embed": "class_embed",
            "sub_bbox_embed": "bbox_embed",
        },

        WEIGHTS_FIXED=[
            "backbone",
            # "transformer.encoder",
            # "transformer.decoder",
        ],

        # detection pretrain weights
        WEIGHTS="/cluster/work/cvl/denfan/diandian/SGTR/weights_backbone/detr_oiv6.pth",

        # TEST_WEIGHTS="/home/allen/MI-projects/SGTR/weights_test/sgtr_oiv6_new/model_0107999.pth",
        TEST_WEIGHTS="/cluster/work/cvl/denfan/diandian/SGTR/CVPODS_output/2023-06-15_13-53-4dor-SGTR-rel_dec-6/model_final.pth",

        RESNETS=dict(
            DEPTH=101,
            OUT_FEATURES=["res5"],
        ),

        DETR=dict(  # entities DETR
            IN_FEATURES="res5",
            # NUM_CLASSES=601,  # for OIv6
            NUM_CLASSES=11,  # for 4D-OR
        ),

        REL_DETR=dict(  # relationship DETR
            USE_GT_ENT_BOX=False,
            TRANSFORMER=dict(
                D_MODEL=256,
                N_HEAD=8,
                SHARE_ENC_FEAT_LAYERS=-1,
                NUM_ENC_LAYERS=3,  # set None will share the encoder with the entities part
                NUM_DEC_LAYERS=rel_dec_layer,
                DIM_FFN=2048,
                DROPOUT_RATE=0.1,
                ACTIVATION="relu",
                PRE_NORM=True,
                RETURN_INTERMEDIATE_DEC=True,
            ),


            ENTITIES_AWARE_HEAD=dict(
                ENTITIES_AWARE_MATCHING=True,  # gt pred matching in training
                ENTITIES_AWARE_RANKING=True,  # pred entities pred relationship matching and ranking in test.
                CROSS_DECODER=True,
                ENABLED=True,


                # SGTR
                USE_ENTITIES_PRED=False,

                INTERACTIVE_REL_DECODER=dict(
                    ENT_DEC_EACH_LVL=True,
                    UPDATE_QUERY_BY_REL_HS=False,
                ),

                ENTITIES_INDEXING=True,
                USE_ENTITIES_INDEXING_RANKING=False,
                INDEXING_TYPE="rule_base",  # feat_att, pred_att rule_base
                INDEXING_TYPE_INFERENCE="rule_base",  # rel_vec
                
                INDEXING_FOCAL_LOSS=dict(
                    ALPHA=0.8,
                    GAMMA=0.0,
                ),


                NUM_FUSE_LAYER=ent_dec_layer,  # for cross encoder

                NUM_DEC_LAYERS=ent_dec_layer,

                ENT_CLS_LOSS_COEFF=0.3,
                ENT_BOX_L1_LOSS_COEFF=0.3,  
                ENT_BOX_GIOU_LOSS_COEFF=1.,
                ENT_INDEXING_LOSS_COEFF=0.0,

                COST_ENT_CLS=0.5,
                COST_BOX_L1=0.6,
                COST_BOX_GIOU=1.25,
                COST_INDEXING=0.00,
                COST_FOREGROUND_ENTITY=0.1,

                REUSE_ENT_MATCH=False,

                USE_REL_VEC_MATCH_ONLY=False,

            ),

            NUM_PRED_EDGES=1,

            NO_AUX_LOSS=False,
            USE_FINAL_MATCH=False,
            USE_SAME_MATCHER=True,

            AUX_LOSS_WEIGHT=aux_loss_weight,

            NUM_QUERIES=60,   # 180    4D-OR has less number of objects

            COST_CLASS=1.0,
            COST_REL_VEC=1.0,

            CLASS_LOSS_COEFF=1.0,
            REL_VEC_LOSS_COEFF=1.0,

            EOS_COEFF=0.08,  # Relative classification weight of the no-object class
            OVERLAP_THRES=0.8,
            NUM_ENTITIES_PAIRING=3,
            # NUM_ENTITIES_PAIRING_TRAIN=40,
            NUM_ENTITIES_PAIRING_TRAIN=25,
            NUM_MAX_REL_PRED=4096,
            MATCHING_RANGE=4096,

            NUM_MATCHING_PER_GT=1,

            DYNAMIC_QUERY=True,
            DYNAMIC_QUERY_AUX_LOSS_WEIGHT=None,

            NORMED_REL_VEC_DIST=False,
            FOCAL_LOSS=dict(ENABLED=False, ALPHA=0.25, GAMMA=2.0, ),
        ),


        ROI_RELATION_HEAD=dict(
            # NUM_CLASSES=30,  # for OIV6
            NUM_CLASSES=14,  # for 4D-OR
            ENABLED=True,
            DATA_RESAMPLING=dict(
                ENABLED=False,
                METHOD="bilvl",
                REPEAT_FACTOR=0.13,
                INSTANCE_DROP_RATE=1.5,
                REPEAT_DICT_DIR=None,
            ),

            # LONGTAIL_PART_DICT=[None, 'h', 'h', 'h', 't', 'b', 't', 't', 'h', 'h', 'b', 't', 't', 't', 'b', 't', 't',
                                # 'h', 't', 'b', 't', 'b', 'b', 't', 't', 'b', 't', 'b', 't', 'h', 't'],   # long tail 问题需要确认一下   h, b, t means head, body, tail.

              LONGTAIL_PART_DICT=[None, 't', 't', 't', 'h', 't', 't', 't', 'b', 'b', 'b', 'b', 't', 't', 'b'], # refer to 4D-OR

        ),
        PROPOSAL_GENERATOR=dict(
            FREEZE=False,
        ),
    ),
    DATASETS=dict(
        # TRAIN=("oi_v6_train",),
        # TEST=("oi_v6_val",),
        TRAIN=("or_train",),
        TEST=("or_val",),
        FILTER_EMPTY_ANNOTATIONS=True,
        FILTER_NON_OVERLAP=False,
        FILTER_DUPLICATE_RELS=True

    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                # ("ResizeShortestEdge", dict(
                #     short_edge_length=[480, 512, 544, 576, 608, 640, 672, 704, 720],
                #     max_size=1000, sample_style="choice")),
                # ("RandomFlip", dict()),
                ("ResizeShortestEdge", dict(
                    short_edge_length=[456],
                    max_size=800, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=456, max_size=800, sample_style="choice")),
            ],
        )
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            # NAME="WarmupMultiStepLR",
            # GAMMA=0.1,
            # MAX_EPOCH=None,
            # MAX_ITER=2.4e5,
            # WARMUP_ITERS=300,
            # STEPS=(6e4, 1.1e5),
            NAME="WarmupMultiStepLR",
            GAMMA=0.1,
            MAX_EPOCH=None,
            MAX_ITER=52000,     # 4d-or train_num:4024, 4024/4=503     503*50=25150  ==> 26000
            WARMUP_ITERS=600,
            STEPS=(10000, 24000),
        ),
        OPTIMIZER=dict(
            NAME="DETRAdamWBuilder",
            BASE_LR=1e-4,
            BASE_LR_RATIO_BACKBONE=1e-4,
            WEIGHT_DECAY=1e-4,
            BETAS=(0.9, 0.999),
            EPS=1e-08,
            AMSGRAD=False,
        ),
        CLIP_GRADIENTS=dict(
            ENABLED=True, CLIP_VALUE=0.1, CLIP_TYPE="norm", NORM_TYPE=2.0,
        ),
        # IMS_PER_BATCH=24,  # 四卡时候的batchsize
        # IMS_PER_DEVICE=6,
        IMS_PER_BATCH=16,  # 2卡时候的batchsize
        IMS_PER_DEVICE=4,
        CHECKPOINT_PERIOD=5000,
    ),
    TEST=dict(
        EVAL_PERIOD=3000,
        RELATION=dict(MULTIPLE_PREDS=False, IOU_THRESHOLD=0.5, EVAL_POST_PROC=True, ),
    ),
    # OUTPUT_DIR=curr_folder.replace(
    #     cvpods_home, os.getenv("CVPODS_OUTPUT")
    # ),
    OUTPUT_DIR="/cluster/work/cvl/denfan/diandian/SGTR/CVPODS_output",
    GLOBAL=dict(
        DUMP_TEST=True,
        LOG_INTERVAL=100,
    ),

)


class OneStageRelDetrConfig(OneStageRelDetrBASEConfig):
    def __init__(self):
        super(OneStageRelDetrConfig, self).__init__()
        self._register_configuration(_config_dict)


config = OneStageRelDetrConfig()
