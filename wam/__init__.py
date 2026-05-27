# coding=utf-8
"""WAM 可安装/可导入包根：网络、训练循环、Runner、采样均在此子树内。

目录::

    wam/
      models/     # 权重、encoder、``mot`` 策略与 ``build_world_stack``
      trainers/   # Accelerate、``TrainModule``、Dataset、增广
      runners/    # ``FMPRunner``、噪声时间步调度等
      samples/    # 训练期采样与指标

仓库根 ``WAM/`` 仍保留 ``main_world.py``、``configs/``、``data/``、``model_config/`` 等入口与配置。
"""
