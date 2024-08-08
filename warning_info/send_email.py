# do what you say,say what you do
# encoding: utf-8
# @author: xulai
# @file: == send_email.py ==
# @time: 2024-04-22 9:00
# @Describe:  发送预警邮件
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import configparser
import os
from . import CONFIG_PATH  # 从__init__.py导入CONFIG_PATH


def get_email_info():
    config = configparser.ConfigParser()
    try:
        # 使用绝对路径读取config.ini文件
        config.read(str(CONFIG_PATH / 'config.ini'), encoding="utf-8")
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

    # 获取指定节中的配置项，这里需要确保这些键在config.ini中确实存在
    try:
        email_username = config['email']['username']
        email_password = config['email']['password']
        smtp_server = config['email']['smtp_server']
        smtp_port = config.getint('email', 'smtp_port')  # 使用getint获取整数值
    except Exception as e:
        print(f"Error retrieving config values: {e}")
        return None

    return email_username, email_password, smtp_server, smtp_port


# 发送邮箱文件
def send(receivers="2360992852@qq.com"):  # 修改为接收者列表
    print(receivers)
    email_info = get_email_info()
    if email_info is None:
        return "Failed to get email info"

    sender, password, host, port = email_info

    message = MIMEMultipart('related')
    message['From'] = sender
    message['To'] = receivers
    message["Subject"] = "火焰检测预警信息"
    message.attach(MIMEText("检测到有火焰和烟雾，请及时处理！"))

    # folder_path = "../result"
    folder_path = str(CONFIG_PATH / '..' / 'result')  # 使用相对路径从包的根路径
    # 遍历文件夹中的图片文件
    count = 0
    for img_filename in os.listdir(folder_path):
        count += 1
        if count % 2 != 0:  # 只发送一半的图片
            continue
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with open(os.path.join(folder_path, img_filename), "rb") as img_file:
                    img_data = img_file.read()
                    img = MIMEImage(img_data, name=img_filename)
                    img.add_header('Content-ID', f'<{img_filename}>')
                    message.attach(img)
            except IOError as e:
                print(f"Error attaching image {img_filename}: {e}")

    try:
        server = smtplib.SMTP(host, port)
        server.login(sender, password)
        server.sendmail(sender, receivers, message.as_string())
        server.quit()
        return "邮件发送成功"
    except smtplib.SMTPException as e:
        return f"邮件发送失败: {e}"

