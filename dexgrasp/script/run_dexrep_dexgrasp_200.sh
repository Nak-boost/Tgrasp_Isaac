CUDA_VISIBLE_DEVICES=0 \
python train.py \
--task=ShadowHandGraspDexRepDexgrasp \
--algo=ppo1 \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/dexrep_dexgrasp_200_test \
--headless \
--num_objs=200
#--test


