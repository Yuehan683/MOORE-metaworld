@echo off
REM ================================
REM Windows version of run_minigrid_ppo_st.sh
REM Usage:
REM   run_minigrid_ppo_st.bat MiniGrid-DoorKey-6x6-v0
REM ================================

REM 切换到项目根目录（等价于 cd ../../../）
cd /d %~dp0\..\..\..

REM 读取第一个参数作为 ENV_NAME
set ENV_NAME=%1

if "%ENV_NAME%"=="" (
    echo ERROR: ENV_NAME is not provided.
    echo Usage: run_minigrid_ppo_st.bat ENV_NAME
    exit /b 1
)

python run_minigrid_ppo_st.py ^
    --n_exp 30 ^
    --env_name %ENV_NAME% ^
    --exp_name ppo_st_baseline ^
    --n_epochs 100 ^
    --n_steps 2000 ^
    --n_episodes_test 16 ^
    --train_frequency 2000 ^
    --lr_actor 1e-3 ^
    --lr_critic 1e-3 ^
    --critic_network MiniGridPPONetwork ^
    --critic_n_features 128 ^
    --actor_network MiniGridPPONetwork ^
    --actor_n_features 128 ^
    --batch_size 256 ^
    --gamma 0.99 
pause
