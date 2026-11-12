"""BiTidy bimanual SO-101 expert tasks: T1 cube handover, T2 bar lift, T3 bar + cube, T4 lid."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.bimanual"

gym.register(
    id="BiTidy-Handover-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:BiHandoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiHandoverPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-HandoverRecv-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.recv_env_cfg:BiHandoverRecvEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiHandoverRecvPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-BarLift-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.bar_env_cfg:BiBarLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiBarLiftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-BarLiftHard-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.bar_hard_env_cfg:BiBarLiftHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiBarLiftHardPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-Stack-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.stack_env_cfg:BiStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiStackPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-StackLeft-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.stack_left_env_cfg:BiStackLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiStackLeftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-StackRecv-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.stack_recv_env_cfg:BiStackRecvEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiStackRecvPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-SortLeft-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.sort_env_cfg:SortLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiSortLeftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-SortRight-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.sort_env_cfg:SortRightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiSortRightPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-SortHandover-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.sort_env_cfg:SortHandoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiSortHandoverPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="BiTidy-Lid3-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.lid3_env_cfg:BiLid3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiLid3PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

for _name, _module, _cfg, _agent in (
    ("LidOpen", "lid_open_env_cfg", "LidOpenEnvCfg", "BiLidOpenPPORunnerCfg"),
    ("BoxPlace", "box_place_env_cfg", "BoxPlaceEnvCfg", "BiBoxPlacePPORunnerCfg"),
    ("LidPlace", "lid_place_env_cfg", "LidPlaceEnvCfg", "BiLidPlacePPORunnerCfg"),
    ("LidSort", "lid_sort_env_cfg", "LidSortEnvCfg", "BiLidSortPPORunnerCfg"),
    ("CupBasket", "cup_basket_env_cfg", "CupBasketEnvCfg", "BiCupBasketPPORunnerCfg"),
):
    for _suffix, _entry in (("", f"{_PACKAGE}.{_module}:{_cfg}"), ("-Render", f"{_PACKAGE}.render:{_name}RenderEnvCfg")):
        gym.register(
            id=f"BiTidy-{_name}-SO101{_suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": _entry, "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:{_agent}", "default_agent": "rsl_rl"},
        )

gym.register(
    id="BiTidy-Lid-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.lid_env_cfg:BiLidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:BiLidPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

for _name, _agent in (
    ("Handover", "BiHandoverPPORunnerCfg"),
    ("HandoverRecv", "BiHandoverRecvPPORunnerCfg"),
    ("BarLift", "BiBarLiftPPORunnerCfg"),
    ("BarLiftHard", "BiBarLiftHardPPORunnerCfg"),
    ("Stack", "BiStackPPORunnerCfg"),
    ("StackLeft", "BiStackLeftPPORunnerCfg"),
    ("StackRecv", "BiStackRecvPPORunnerCfg"),
    ("SortLeft", "BiSortLeftPPORunnerCfg"),
    ("SortRight", "BiSortRightPPORunnerCfg"),
    ("SortHandover", "BiSortHandoverPPORunnerCfg"),
    ("Lid3", "BiLid3PPORunnerCfg"),
    ("Lid", "BiLidPPORunnerCfg"),
):
    gym.register(
        id=f"BiTidy-{_name}-SO101-Render",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_PACKAGE}.render:{_name}RenderEnvCfg",
            "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents:{_agent}",
            "default_agent": "rsl_rl",
        },
    )
