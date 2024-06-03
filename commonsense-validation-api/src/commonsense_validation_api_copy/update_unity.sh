#!/usr/bin/env bash
rsync -a ./*.py unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/src/commonsense_validation_api
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/xmeans_sweep.yaml unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/gmeans_sweep.yaml unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/hac_sweep.yaml unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/slurm_wandb_agent.py unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/gt_sweep.yaml unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/
rsync -a /Users/lorraine/UMass/2022/Research/cfc/evaluator/commonsense-validation-api/analysis_result/top.py unity:/work/xiangl_umass_edu/cfc/commonsense-validation-api/src/scores/