"""仅训练 Video Expert（Wan）入口：命令行与 ``main_world`` 一致，训练逻辑见 ``wam.trainers.train_video``。"""

from accelerate.logging import get_logger

from main_world import parse_args
from wam.trainers.train_video import train


if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)
