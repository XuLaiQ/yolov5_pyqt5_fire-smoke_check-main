# do what you say,say what you do
# encoding: utf-8
# @author: xulai
# @file: == main.py ==
# @time: 2024-04-23 15:00
# @Describe: 火焰烟雾识别预警

from pyqt5 import fire
from PyQt5 import QtGui
from PyQt5.QtCore import QDateTime, QTimer, QThread, pyqtSignal
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QLabel, QMainWindow,
                             QGridLayout, QScrollArea, QMessageBox)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

# 引入YOLOv5的pt模型部署类  多线程配置
import argparse
import numpy as np
import os
import sys
from pathlib import Path
import torch
from datetime import datetime
from warning_info.send_email import send  # 发送邮箱
from warning_info.warning_sound import AlarmSoundThread  # 警报声
import threading
import re

from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# YOLOv5的多线程管理
class ModelThread(QThread):
    # 信号状态
    send_img = pyqtSignal(np.ndarray)
    results = pyqtSignal(str)
    send_msg = pyqtSignal(str)  # emit：detecting/pause/stop/finished/error msg

    check_back = pyqtSignal(np.ndarray)  # 返回检测后的图片数组
    show_popup_signal = pyqtSignal()  # 弹窗进程
    play_alarm_signal = pyqtSignal()  # 警报声进程

    def __init__(self):
        super().__init__()
        # YOLOv5参数
        self.opt = None
        self.check_stop = False  # 停止检测
        self.email_account = ''  # 邮箱账号
        self.pop = 0  # 控制弹窗进程
        self.sound = 0  # 控制警报声进程

        self.init_parameter()

    # 检测模型参数初始化
    def init_parameter(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/fires.pt', help='onnx path(s)')
        parser.add_argument('--source', type=str, default=0,
                            help='file/dir/URL/glob/screen/0(webcam)/')  # ROOT / 'data/images'
        self.opt = parser.parse_args()

    # 检验邮箱格式是否正确
    @staticmethod
    def is_valid_email(email):
        # 定义一个正则表达式，用于匹配多种邮箱格式
        email_pattern = r'''
            ^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+       # 用户名（允许字母、数字、特殊字符）
            @[a-zA-Z0-9-]+                           # 域名（允许字母、数字、短划线）
            (\.[a-zA-Z0-9-]+)*                       # 域名后缀（允许字母、数字、短划线，可能多个）
            \.[a-zA-Z]{2,}$                          # 顶级域名（至少两个字母）
        '''
        # 使用re模块的match方法来检查邮箱是否匹配
        return bool(re.match(email_pattern, email, re.VERBOSE))

    def run(self,
            data=ROOT / '',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            project=ROOT / 'result',  # save results to project/name
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            ):
        source = str(self.opt.source)
        print(source)
        weights = self.opt.weights
        self.results.emit("使用模型为：" + weights)
        webcam = source.isnumeric() or source.endswith('.streams')
        # 加载模型
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # 数据加载
        bs = 1  # batch_size
        if webcam:  # 摄像头
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        else:  # 视频和图片
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        count = 0  # 控制保存的图片
        try:
            # 进行推理
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                # Inference
                with dt[1]:
                    pred = model(im, augment=augment, visualize=False)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        im0 = cv2.flip(im0, 1)  # 摄像头镜像
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    check_str = ""  # 检测结果
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        # 打印结果
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            check_str += f"{n} {names[int(c)]}, "  # add to string
                        # 画框
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # Stream results
                    im0 = annotator.result()
                    self.send_img.emit(im0)

                    os.makedirs(project, exist_ok=True)
                    flag = len(os.listdir(project))
                    # 保存检测到的图片
                    if len(det) and flag <= 20 and webcam or vid_cap:  # flag最多保存20张照片；
                        count += 1
                        if count % 20 == 0:  # count表示每个120帧保存一张
                            # 获取当前时间
                            now = datetime.now()
                            # 格式化时间字符串，例如：'2024-04-11_15-30-00'
                            time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                            # 定义文件扩展名
                            extension = '.jpg'
                            # 拼接完整的文件名，包括路径、时间字符串和扩展名
                            output_filepath = os.path.join(project, f"camera_{time_str}{extension}")
                            # 复制图像
                            new_array = im0.copy()
                            # 获取图像的尺寸
                            height, width = new_array.shape[:2]
                            # 设置文本位置在图像的左下方
                            text_x = width - (width // 4)  # 可以调整为其他值，例如图像宽度的负值
                            text_y = height - 20  # 20 是文本底部与图像底部的距离，可以根据实际需要调整
                            cv2.putText(new_array, time_str, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 250, 154), thickness=1)
                            cv2.imwrite(output_filepath, new_array)

                    if flag >= 10:
                        # 在需要显示弹窗的地方发出信号
                        if self.pop == 0:
                            self.show_popup_signal.emit()  # 而不是直接调用 show_popup 方法
                            self.pop += 1
                            print("发出显示弹窗信号")

                        # 检验邮箱格式
                        if self.is_valid_email(self.email_account):
                            m = send(self.email_account)  # 发送预警邮箱
                            self.results.emit(m)

                        # 列出目录下的所有文件和文件夹
                        entries = os.listdir(project)
                        # 遍历列表中的所有条目
                        for entry in entries:
                            # 构建完整的文件路径
                            full_path = os.path.join(project, entry)
                            # 检查文件是否是图片，基于扩展名判断
                            if full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                                # 删除文件
                                os.remove(full_path)

                # 打印检测结果和时间
                self.results.emit(f"{check_str}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
                if self.check_stop:
                    # 摄像头检测的时候关闭摄像头
                    if webcam:
                        dataset.close_streams()
                    # 摄像头和视频检测的视频才会进行图片保存
                    if len(os.listdir(project)) > 0 and webcam or vid_cap:
                        self.results.emit(f"检测图片已经保存到 {project} 文件夹")
                    break

        except Exception as e:
            self.send_msg.emit(f'Error: {e}')
            self.check_stop = True
        finally:
            # 关闭所有OpenCV窗口
            cv2.destroyAllWindows()


# 界面逻辑部分
class FireWin(QMainWindow, fire.Ui_MainWindow):
    def __init__(self):
        super(FireWin, self).__init__()
        self.setupUi(self)
        # 控制摄像头线程
        self.model_th = None
        # 宽，高
        self.width = self.camera.width()
        self.height = self.camera.height()

        self.timer_camera = QTimer(self)

        self.print_result = self.result_output  # 结果打印
        self.email_ = self.lineEdit  # 得到接收邮箱
        self.image_viewer = None  # 展示自动保存的检测截图
        self.model_type = None  # 模型选择
        self.result_number = 0  # 记录打印结果的条数（控制打印条数）
        self.play_time = 1000000000000  # 警报时间最初设置为1000000000000s,只有点击弹窗上的OK按钮后才能停止

        # search models automatically
        self.comboBox_2.clear()
        self.pt_list = os.listdir(os.path.join(ROOT, "weights"))
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize(os.path.join(ROOT, "weights") + '/' + x), reverse=True)
        self.comboBox_2.clear()
        self.comboBox_2.addItems(self.pt_list)
        self.comboBox_2.currentIndexChanged[int].connect(self.search_pt)

        self.init_time()
        self.slot_init()
        self.init_thread()

    # 多线程
    def init_thread(self):
        self.model_th = ModelThread()
        self.model_type = self.comboBox_2.currentText()
        self.model_th.opt.weights = os.path.join(ROOT, "weights") + "/%s" % self.model_type
        self.model_th.opt.source = '0'  # 打开摄像头
        self.model_th.check_back.connect(self.check_camera)
        self.model_th.send_msg.connect(self.statusbar_massage)
        self.model_th.send_img.connect(self.check_camera)
        self.model_th.results.connect(self.add_result)
        self.model_th.show_popup_signal.connect(self.on_show_popup)  # 新增信号连接，弹窗

    # 时间的显示
    def init_time(self):
        self.status_show_time()

    # 信号和槽函数绑定
    def slot_init(self):
        self.pushButton.clicked.connect(self.button_check_camera)
        self.look.clicked.connect(self.show_images)
        self.timer_camera.timeout.connect(self.show_camera)
        self.clear_all.clicked.connect(self.clear_result)
        self.picture.clicked.connect(self.select_media_file)

    # 打开摄像头
    def button_check_camera(self):
        if not self.timer_camera.isActive():
            self.print_result.append("正在打开摄像头")
            self.pushButton.setText(u'停止检测')
            self.camera.setText("视频加载中...")
            self.statusbar.showMessage("打开视频")
            self.timer_camera.start(30)
            self.model_th.opt.source = '0'  # 打开摄像头 0表示笔记本自带摄像头  1表示外接摄像头
            self.model_th.email_account = self.email_.text()  # 获取到输入的邮箱账号
            self.model_th.check_stop = False
            self.model_th.start()

        else:
            self.model_th.check_stop = True
            # 等待线程安全地结束
            self.model_th.wait()  # 确保线程已经停止
            # 停止定时器
            self.timer_camera.stop()
            self.pushButton.setText(u'开始检测')
            self.statusbar.showMessage("关闭相机")
            self.print_result.append("正在关闭摄像头")

    # 视频检测画面
    def check_camera(self, image=None):
        if image is not None:
            # opencv 默认图像格式是rgb QImage要使用BRG
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            show_image = QtGui.QImage(show.data,
                                      show.shape[1],
                                      show.shape[0],
                                      QtGui.QImage.Format_RGB888)
            jpg_out = QtGui.QPixmap(show_image).scaled(self.camera.width(), self.camera.height(), Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation)
            self.camera.setPixmap(jpg_out)

    # 展示视频画面
    def show_camera(self):
        self.check_camera()

    # 图片检测/视频检测
    def check_picture_video(self, path):
        self.model_th.email_account = self.email_.text()  # 获取到输入的邮箱账号
        self.model_th.opt.source = path
        self.model_th.check_stop = False
        self.model_th.start()

    # 本地上传(图片/视频)
    def select_media_file(self):
        self.model_th.check_stop = True
        self.camera.clear()
        self.camera.setText("显示区域")

        # 弹出文件选择对话框
        filename, _ = QFileDialog.getOpenFileName(self, "选择媒体文件", './test',
                                                  "Image files (*.jpg *.jpeg *.png *.gif *.webp "
                                                  "*.mp4 *.avi *.mkv *.mov *.flv *.wmv)")
        # 如果用户选择了文件
        if filename:
            self.check_picture_video(filename)

    # 弹窗和警报声
    def on_show_popup(self):
        stop_event = threading.Event()  # 创建一个事件对象
        a = AlarmSoundThread(self.play_time, stop_event)
        a.start()
        reply = QMessageBox.warning(self, "警告", "检测到火焰烟雾，请及时处理！", QMessageBox.Ok)
        if reply == QMessageBox.Ok:
            stop_event.set()  # 设置事件，这将导致线程检查这个事件并在适当的时候停止
            self.model_th.pop -= 1  # 用户点击确认后重置计数器

    # 查看所有保存的照片
    def show_images(self):
        folder_path = os.path.join(ROOT, "result")
        os.makedirs(folder_path, exist_ok=True)
        # 指定文件夹目录
        flag = len(os.listdir(folder_path))
        if flag:
            # 创建新窗口并显示图片
            self.image_viewer = ImageViewer(folder_path)
            self.image_viewer.show()
        else:
            self.result_output.append("未检测到火焰/烟雾，文件夹中没有图片")

    # 选择模型
    def search_pt(self):
        pt_list = os.listdir(os.path.join(ROOT, "weights"))
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize(os.path.join(ROOT, "weights") + '/' + x), reverse=True)

        self.model_type = self.comboBox_2.currentText()
        self.model_th.opt.weights = os.path.join(ROOT, "weights") + "/%s" % self.model_type

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox_2.clear()
            self.comboBox_2.addItems(self.pt_list)

    # 清空所有打印的结果
    def clear_result(self):
        self.print_result.clear()
        self.result_number = 0

    # 打印检测结果
    def add_result(self, res):
        self.result_number += 1
        # 最多打印500条就清空
        if self.result_number >= 500:
            self.clear_result()
        self.print_result.append(str(res))

    # 在底部展示错误信息
    def statusbar_massage(self, msg):
        self.statusbar.showMessage("错误信息：" + msg)

    # 关闭窗口
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"确定要关闭吗？")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    # 显示实时时间
    def status_show_time(self):
        timer = QTimer(self)  # 自定义QTimer类
        timer.start(1000)  # 每1s运行一次
        timer.timeout.connect(self.update_time)  # 与updateTime函数连接

    def update_time(self):
        time = QDateTime.currentDateTime()  # 获取现在的时间
        timeplay = time.toString('yyyy-MM-dd hh:mm:ss dddd')  # 设置显示时间的格式
        self.timedisplay.setText(timeplay)  # 设置timeLabel控件显示的内容


# 查看保存的图片
class ImageViewer(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.setWindowTitle('图片查看器')
        self.folder_path = folder_path
        self.images = [os.path.join(self.folder_path, img) for img in os.listdir(self.folder_path) if
                       img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # 创建滚动区域和网格布局
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.widget = QWidget()
        self.resize(650, 500)
        self.layout = QGridLayout(self.widget)

        # 加载并显示所有图片
        for i, image_path in enumerate(self.images):
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(600, 500, Qt.KeepAspectRatio)  # 缩放图片以适应大小
            label = QLabel(self.widget)
            label.setPixmap(pixmap)
            # 将图片添加到网格布局中，每行只有一列
            self.layout.addWidget(label, i, 0)

        self.scroll_area.setWidget(self.widget)
        self.setCentralWidget(self.scroll_area)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FireWin()
    ui.setWindowIcon(QIcon(os.path.join(ROOT, './pyqt5/components/win_fire.png')))
    ui.show()
    sys.exit(app.exec_())