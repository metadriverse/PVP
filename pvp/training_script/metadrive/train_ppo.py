import copy
from ray import tune
from pvp.utils.expert_common import SaverCallbacks
from pvp.utils.expert_human_in_the_loop_env import HumanInTheLoopEnv
from pvp.utils.train_eval_config import baseline_eval_config
from pvp.utils.rllib_utils import get_train_parser
from pvp.utils.rllib_utils.train import train
from ray.rllib.agents.ppo.ppo import PPOTrainer

evaluation_config = {"env_config": copy.deepcopy(baseline_eval_config)}

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "PPO"
    stop = {"timesteps_total": 1000_0000}

    config = dict(
        env=HumanInTheLoopEnv,
        env_config=dict(main_exp=False),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=30,

        # ===== Training =====
        horizon=1500,
        num_sgd_iter=20,
        lr=5e-5,
        grad_clip=10.0,
        rollout_fragment_length=256,
        sgd_minibatch_size=100,
        train_batch_size=30000,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.5,
        num_workers=8,
        clip_actions=False
    )

    train(
        PPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=10,
        custom_callback=SaverCallbacks,
        # test_mode=True,
        # local_mode=True
        wandb_key_file="~/wandb_api_key_file.txt",
        wandb_project="old_2022",
    )
