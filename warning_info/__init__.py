# do what you say,say what you do
# encoding: utf-8
# @author: xulai
# @file: == __init__.py.py ==
# @time: 2024-04-21 16:29
# @Describe: 这个包是实现预警的功能
#  当检测到特定目标时，你可以实现预警功能。预警可以多种方式实现，如：
#        显示警告文本：在检测到的目标周围显示警告文本。
#        播放警告声音：使用pygame库播放警告声音。
#        发送通知：通过电子邮件、短信或其他方式发送通知。
#        保存帧：将包含检测目标的帧保存为图像或视频。
from pathlib import Path

# 获取当前包的路径
PACKAGE_PATH = Path(__file__).parent

# 设置config.ini文件的全局路径
CONFIG_PATH = PACKAGE_PATH
