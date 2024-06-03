#!/usr/bin/env bash
rsync -a ./*.py gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/src/commonsense_validation_api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/xmeans_sweep.yaml gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/gmeans_sweep.yaml gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/hac_sweep.yaml gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/slurm_wandb_agent.py gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/gt_sweep.yaml gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/analysis_result/top.py gypsum-remote:/mnt/nfs/work1/mccallum/xiangl/2022_cfc/evaluator/commonsense-validation-api/src/scores/
