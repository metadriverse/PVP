"""
Training script for training PVP in MiniGrid environment.
"""
import argparse
import os
from pathlib import Path

# import gym
import gymnasium as gym
import torch
from minigrid.wrappers import ImgObsWrapper

from pvp.experiments.minigrid.minigrid_env import MinigridWrapper
from pvp.experiments.minigrid.minigrid_model import MinigridCNN
from pvp.pvp_dqn import PVPDQN
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.dqn.policies import CnnPolicy
from pvp.utils.utils import get_time_str
import minigrid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="pvp_minigrid", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    # parser.add_argument(
    #     "--device",
    #     required=True,
    #     choices=['wheel', 'gamepad', 'keyboard'],
    #     type=str,
    #     help="The control device, selected from [wheel, gamepad, keyboard]."
    # )

    parser.add_argument("--env_name", default="MiniGrid-Empty-Random-6x6-v0", type=str, help="Name of Gym environment")
    # Or use environment: --env-name MiniGrid-MultiRoom-N6-v0

    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    args = parser.parse_args()

    # ===== Set up some arguments =====
    experiment_batch_name = args.exp_name
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, 'keyboard', get_time_str())

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    env_name = args.env_name
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    eval_log_dir = trial_dir / "evaluations"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(),

        # Algorithm config
        algo=dict(
            policy=CnnPolicy,
            policy_kwargs=dict(features_extractor_class=MinigridCNN, activation_fn=torch.nn.Tanh, net_arch=[
                64,
            ]),

            # === HACO setting ===
            replay_buffer_kwargs=dict(discard_reward=True  # PZH: We run in reward-free manner!
                                      ),
            exploration_fraction=0.0,  # 1% * 100k = 1k
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            env=None,
            optimize_memory_usage=True,

            # Hyper-parameters are collected from https://arxiv.org/pdf/1910.02078.pdf
            # MiniGrid specified parameters
            buffer_size=10_000,
            learning_rate=1e-4,

            # === New hypers ===
            learning_starts=50,  # PZH: Original DQN has 100K warmup steps
            batch_size=256,  # or 32?
            train_freq=1,  # or 4?
            tau=0.005,
            target_update_interval=1,
            # target_update_interval=50,

            # === Old DQN hypers ===
            # learning_starts=1000,  # PZH: Original DQN has 100K warmup steps
            # batch_size=32,  # Reduce the batch size for real-time copilot
            # train_freq=4,
            # tau=1.0,
            # target_update_interval=1000,
            gradient_steps=32,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Meta data
        project_name=project_name,
        team_name=team_name,
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    minigrid.register_minigrid_envs()
    env = gym.make(env_name)
    env = MinigridWrapper(env, enable_render=True, enable_human=True)
    env = Monitor(env=env, filename=str(trial_dir))
    env = ImgObsWrapper(env)
    train_env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=4)

    # ===== Also build the eval env =====
    def _make_eval_env():
        env = gym.make(env_name)
        env = MinigridWrapper(env, enable_render=False, enable_human=False)
        env = Monitor(env=env, filename=eval_log_dir)
        env = ImgObsWrapper(env)
        return env

    eval_env = VecFrameStack(DummyVecEnv([_make_eval_env]), n_stack=4)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    save_freq = 500  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    # TODO: Do we have similar 'stop td at intervention start' thing here?
    model = PVPDQN(**config["algo"])

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=20,
        n_eval_episodes=10,
        eval_log_path=trial_dir,

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
    )
