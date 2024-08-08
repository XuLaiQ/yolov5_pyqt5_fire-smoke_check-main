# do what you say,say what you do
# encoding: utf-8
# @author: xulai
# @file: == warning_sound.py ==
# @time: 2024-04-23 17:01
# @Describe:  发出警报声
from . import CONFIG_PATH  # 从__init__.py导入CONFIG_PATH
import threading
import pygame
import os


class AlarmSoundThread(threading.Thread):
    def __init__(self, sound_time, stop_event):
        super().__init__()
        self.set_sound_times = sound_time
        self.stop_event = stop_event  # 添加一个事件对象来控制线程的停止

    def run(self):
        # 初始化pygame
        pygame.init()

        # 设置混音器的音量（0.0到1.0）
        pygame.mixer.music.set_volume(1.0)

        # 加载音频文件
        audio_path = os.path.join(CONFIG_PATH, 'warning.mp3')
        sound = pygame.mixer.Sound(audio_path)

        # 计算需要循环播放的次数，以确保总播放时间至少为一分钟
        audio_length_seconds = sound.get_length()  # 获取音频文件的时长（秒）
        play_count = int(self.set_sound_times / audio_length_seconds)  # 一分钟除以音频时长，得到循环次数

        # 循环播放音频指定的次数
        for i in range(play_count):
            # 播放音频
            sound.play()

            # 等待音频播放完成
            while pygame.mixer.get_busy() and not self.stop_event.is_set():
                pygame.time.Clock().tick(10)
            # 如果停止事件被设置，则退出循环
            if self.stop_event.is_set():
                break
        # 退出pygame
        pygame.quit()
