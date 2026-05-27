# coding=utf-8
"""**向后兼容** — Wan2.2 视频 Runner 入口。

``train_video.py`` 已不再自动切换使用本模块；
直接使用 ``runner_video.VideoRunner`` 并传 ``video_variant='wan22'`` 即可。
保留本文件仅确保显式引用 ``train_runner_module: runner_video_wan2_2`` 的旧 config 不会报错。
"""

from wam.runners.runner_video import VideoRunner as _VideoRunner

class VideoRunner(_VideoRunner):
    """默认 ``video_variant='wan22'``。推荐直接使用 ``runner_video.VideoRunner``。"""

    def __init__(self, **kwargs):
        kwargs.setdefault("video_variant", "wan22")
        super().__init__(**kwargs)