@echo off
REM 用法：
REM run_metaworld_mt10_mhsac_mt_moore_svd_test.bat 4 42

set N_EXPERTS=%1
set SEED=%2

REM 进入项目根目录（等价于 cd ../../）
cd /d %~dp0\..\..

python run_metaworld_sac_mt_svd.py ^
  --seed %SEED% ^
  --n_exp 1 ^
  --exp_type MT10 ^
  --exp_name mhsac_moore_svd_test_400x3lx%N_EXPERTS%e_gpu ^
  --results_dir logs/metaworld ^
  --batch_size 128 ^
  --n_epochs 5 ^
  --n_steps 10000 ^
  --horizon 150 ^
  --gamma 0.99 ^
  --lr_actor 3e-4 ^
  --lr_critic 3e-4 ^
  --lr_alpha 1e-4 ^
  --log_std_min -10 ^
  --log_std_max 2 ^
  --actor_network MetaworldSACMixtureMHActorNetwork ^
  --critic_network MetaworldSACMixtureMHCriticNetwork ^
  --orthogonal ^
  --n_experts %N_EXPERTS% ^
  --activation Linear ^
  --agg_activation Linear Tanh ^
  --actor_n_features 400 400 400 ^
  --critic_n_features 400 400 400 ^
  --shared_mu_sigma ^
  --initial_replay_size 1000 ^
  --max_replay_size 1000000 ^
  --warmup_transitions 1000 ^
  --n_episodes_test 5 ^
  --train_frequency 1 ^
  --sample_task_per_episode ^
  --rl_checkpoint_interval 1 ^
  --use_cuda

pause