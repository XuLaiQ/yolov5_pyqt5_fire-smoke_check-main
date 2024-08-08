# do what you say,say what you do
# encoding: utf-8
# @author: xulai
# @file: == pop_up.py ==
# @time: 2024-04-24 19:48
# @Describe:  预警弹窗
"""
在主文件main.py中的354行函数：
    # 弹窗和警报声
    def on_show_popup(self):
        stop_event = threading.Event()  # 创建一个事件对象
        a = AlarmSoundThread(self.play_time, stop_event)
        a.start()
        reply = QMessageBox.warning(self, "警告", "检测到火焰烟雾，请及时处理！", QMessageBox.Ok)
        if reply == QMessageBox.Ok:
            stop_event.set()  # 设置事件，这将导致线程检查这个事件并在适当的时候停止
            self.model_th.pop -= 1  # 用户点击确认后重置计数器
    实现
"""
