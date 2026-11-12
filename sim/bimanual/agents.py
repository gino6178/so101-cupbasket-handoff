"""RSL-RL PPO configuration for the bimanual handover state teacher."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import SO101StatePPORunnerCfg


@configclass
class BiHandoverPPORunnerCfg(SO101StatePPORunnerCfg):
    """Same algorithm as the single-arm teacher; wider networks for the 12-D action and the longer task."""

    max_iterations = 3000
    save_interval = 100
    experiment_name = "bitidy_handover_state"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3, std_type="log", std_range=(0.05, 0.4)),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)


@configclass
class BiHandoverRecvPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_handover_recv"


@configclass
class BiBarLiftPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_barlift_state"


@configclass
class BiBarLiftHardPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_barlift_hard_state"


@configclass
class BiStackPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_stack_state"


@configclass
class BiStackLeftPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_stack_left"


@configclass
class BiStackRecvPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_stack_recv"


@configclass
class BiSortLeftPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_sort_left"


@configclass
class BiSortRightPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_sort_right"


@configclass
class BiSortHandoverPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_sort_handover"


@configclass
class BiLid3PPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_lid3"


@configclass
class BiLidOpenPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_lid_open"


@configclass
class BiBoxPlacePPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_box_place"


@configclass
class BiLidPlacePPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_lid_place"


@configclass
class BiLidSortPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_lid_sort"


@configclass
class BiLidPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_lid_state"


@configclass
class BiCupBasketPPORunnerCfg(BiHandoverPPORunnerCfg):
    experiment_name = "bitidy_cup_basket"
