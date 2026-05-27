"""仅训练 Video Expert（Wan）+ Qwen2.5-VL 条件；入口与 ``main_video`` 相同，训练逻辑见 ``wam.trainers.train_video_qwen``。"""

from accelerate.logging import get_logger

from main_world import parse_args
from wam.trainers.train_video_qwen import train


if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)
