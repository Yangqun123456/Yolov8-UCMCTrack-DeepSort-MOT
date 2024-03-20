# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

import json
import sys
import os
import platform
import cv2
# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *
from YoloPredictor import YoloPredictor
from utils.ClickableLabel import ClickableLabel
from utils.DraggableLabel import DraggableLabel
from widgets import *
from utils.capnums import Camera

# FIX Problem for High DPI and Scale above 100%
os.environ["QT_FONT_DPI"] = "96"

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None


class MainWindow(QMainWindow):
    main2yolo_begin_sgl = Signal()  # 主窗口向yolo实例发送执行信号

    def __init__(self):
        QMainWindow.__init__(self)
        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui
        self.reset_splitter(widgets)
        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "MOT"
        description = "MOT-多目标跟踪系统"
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(
            lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # QTableWidget PARAMETERS
        # ///////////////////////////////////////////////////////////////
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////
        widgets.run_button.clicked.connect(self.run_or_continue)
        widgets.stop_button.clicked.connect(self.stop)
        # LEFT MENUS
        widgets.src_file_button.clicked.connect(self.open_src_file)
        widgets.src_cam_button.clicked.connect(self.camera_select)
        widgets.src_rtsp_button.clicked.connect(self.rtsp_seletction)
        widgets.src_lock_button.clicked.connect(self.lock_id_selection)
        # RIGHT MENUS
        widgets.crossing_line_button.clicked.connect(self.crossing_line)
        widgets.region_counter_button.clicked.connect(self.region_counter)
        widgets.heatmap_button.clicked.connect(self.heatmap)
        widgets.speed_estimate_button.clicked.connect(self.speed_estimate)
        widgets.distence_estimate_button.clicked.connect(
            self.distence_estimate)
        # EXTRA LEFT BOX

        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.src_file_button.setStyleSheet(
            UIFunctions.selectMenu(widgets.src_file_button.styleSheet()))

        # 更改模型
        self.pt_list = os.listdir('./weights')
        self.pt_list = [file for file in self.pt_list if file.endswith(
            '.pt') or file.endswith('.engine')]
        self.pt_list.sort(key=lambda x: os.path.getsize(
            './weights/' + x))  # 按文件大小排序
        widgets.model_box.clear()
        widgets.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)  # 计时器：每2秒监视模型文件更改一次
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)
        widgets.model_box.currentTextChanged.connect(self.change_model)
        widgets.trackerMethod.currentTextChanged.connect(
            self.change_tracker_model)
        # 更改置信度
        widgets.conf_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, 'conf_spinbox'))
        widgets.conf_slider.valueChanged.connect(
            lambda x: self.change_val(x, 'conf_slider'))

        # Yolo-v8 线程
        self.yolo_predict = YoloPredictor()  # 实例化yolo检测
        self.select_model = widgets.model_box.currentText()
        self.yolo_predict.new_model_name = "./weights/%s" % self.select_model
        self.yolo_thread = QThread()
        self.yolo_predict.yolo2main_box_img.connect(
            lambda x: self.show_image(x, widgets.res_video))   # 绘制检测框图
        self.yolo_predict.yolo2main_second_img.connect(
            lambda x: self.show_image(x, widgets.second_video))  # 绘制热力图/速度图
        self.yolo_predict.yolo2main_status_msg.connect(
            lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(
            lambda x: widgets.fps_label.setText(x))
        self.yolo_predict.yolo2main_class_num.connect(
            lambda x: widgets.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(
            lambda x: widgets.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(
            lambda x: widgets.progressBar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # 设置统计数值为空
        widgets.Class_num.setText('--')
        widgets.Target_num.setText('--')
        widgets.fps_label.setText('--')

    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event: QMouseEvent):
        self.dragPos = event.globalPosition().toPoint()

    @staticmethod
    def show_image(img_src, label):
        try:
            if label.geometry().width() == 0 or label.geometry().height() == 0:
                return
            if len(img_src.shape) == 3:
                ih, iw, _ = img_src.shape
            if len(img_src.shape) == 2:
                ih, iw = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))
            label.ih = ih
            label.iw = iw
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 选择文件
    def open_src_file(self):
        UIFunctions.resetStyle(self, 'src_file_button')
        widgets.src_file_button.setStyleSheet(
            UIFunctions.selectMenu(widgets.src_file_button.styleSheet()))
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(
            self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    # 开始或继续检测
    def run_or_continue(self):
        if self.yolo_predict.source == '' or self.yolo_predict.source == None:
            self.show_status('请在检测前选择输入源...')
            widgets.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if widgets.run_button.isChecked():
                widgets.run_button.setChecked(True)
                widgets.model_box.setEnabled(False)
                widgets.trackerMethod.setEnabled(False)
                widgets.conf_slider.setEnabled(False)
                widgets.conf_spinbox.setEnabled(False)

                self.show_status('检测中...')
                if '0' in self.yolo_predict.source or 'rtsp' in self.yolo_predict.source:
                    widgets.progressBar.setFormat('实时视频流检测中...')
                if 'avi' in self.yolo_predict.source or 'mp4' in self.yolo_predict.source:
                    widgets.progressBar.setFormat("当前检测进度:%p%")
                self.yolo_predict.continue_dtc = True
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("暂停...")

                widgets.run_button.setChecked(False)

    # 终止检测
    def stop(self):
        try:
            self.yolo_thread.quit()  # 结束线程
        except:
            pass
        self.yolo_predict.stop_dtc = True
        widgets.run_button.setChecked(False)  # 恢复按钮状态
        widgets.model_box.setEnabled(True)  # 恢复模型选择
        widgets.trackerMethod.setEnabled(True)  # 恢复跟踪算法选择
        widgets.conf_slider.setEnabled(True)
        widgets.conf_spinbox.setEnabled(True)
        widgets.res_video.clear()  # 清空视频显示
        widgets.second_video.clear()  # 清空视频显示
        widgets.progressBar.setValue(0)  # 进度条清零
        widgets.Class_num.setText('--')
        widgets.Target_num.setText('--')
        widgets.fps_label.setText('--')
        self.show_status("检测终止")

    # 显示状态信息
    def show_status(self, msg):
        widgets.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            widgets.run_button.setChecked(False)
            widgets.progressBar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 终止线程
        elif msg == 'Detection terminated!' or msg == '检测终止':
            widgets.run_button.setChecked(False)
            widgets.progressBar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 终止线程
            widgets.res_video.clear()
            widgets.Class_num.setText('--')
            widgets.Target_num.setText('--')
            widgets.fps_label.setText('--')

    # 选择摄像头
    def camera_select(self):
        UIFunctions.resetStyle(self, 'src_cam_button')
        widgets.src_cam_button.setStyleSheet(
            UIFunctions.selectMenu(widgets.src_cam_button.styleSheet()))
        self.stop()
        # 获取本地摄像头数量
        _, cams = Camera().get_cam_num()
        popMenu = QMenu()
        popMenu.setFixedWidth(180)
        popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 20px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 212, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 250, 200,50);}
                                            ''')
        for cam in cams:
            exec("action_%s = QAction('%s 号摄像头')" % (cam, cam))
            exec("popMenu.addAction(action_%s)" % cam)
        pos = QCursor.pos()
        action = popMenu.exec(pos)
        if action:
            str_temp = ''
            selected_stream_source = str_temp.join(
                filter(str.isdigit, action.text()))  # 获取摄像头号，去除非数字字符
            self.yolo_predict.source = selected_stream_source
            self.show_status('摄像头设备:{}'.format(action.text()))

    # 选择rtsp
    def rtsp_seletction(self):
        UIFunctions.resetStyle(self, 'src_rtsp_button')
        widgets.src_rtsp_button.setStyleSheet(
            UIFunctions.selectMenu(widgets.src_rtsp_button.styleSheet()))
        self.rtsp_window = Ui_Dialog()
        self.dialog = QDialog()
        self.rtsp_window.setupUi(self.dialog)
        config_file = 'config/ip.json'
        if os.path.exists(config_file):
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.lineEdit.setText(ip)
        self.dialog.show()
        self.rtsp_window.buttonBox.accepted.connect(
            lambda: self.load_rtsp(self.rtsp_window.lineEdit.text()))

        # 加载RTSP

    def load_rtsp(self, ip):
        self.stop()
        self.yolo_predict.source = ip
        new_config = {"ip": ip}
        new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
        with open('config/ip.json', 'w', encoding='utf-8') as f:
            f.write(new_json)
        self.show_status('加载rtsp地址:{}'.format(ip))
        self.dialog.close()
    # 单目标跟踪

    def lock_id_selection(self):
        self.yolo_predict.lock_id = None
        self.id_window = Ui_Dialog()
        self.dialog = QDialog()
        self.id_window.setupUi(self.dialog)
        self.id_window.label.setText('请输入要跟踪的目标ID:')
        config_file = 'config/id.json'
        if os.path.exists(config_file):
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            id = config['id']
        self.id_window.lineEdit.setText(id)
        self.dialog.show()
        self.id_window.buttonBox.accepted.connect(
            lambda: self.set_lock_id(self.id_window.lineEdit.text()))

    def set_lock_id(self, lock_id):
        self.yolo_predict.lock_id = lock_id
        new_config = {"id": lock_id}
        new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
        with open('config/id.json', 'w', encoding='utf-8') as f:
            f.write(new_json)
        self.show_status('加载ID:{}'.format(lock_id))
        self.dialog.close()

    # 模型更换
    def change_model(self, x):
        self.select_model = widgets.model_box.currentText()
        self.yolo_predict.new_model_name = "./weights/%s" % self.select_model
        self.show_status('更改模型：%s' % self.select_model)

    # 跟踪算法更换
    def change_tracker_model(self, x):
        self.yolo_predict.tracker = widgets.trackerMethod.currentText()
        self.show_status('更改跟踪算法：%s' % widgets.trackerMethod.currentText())

    # 循环监测文件夹的文件变化
    def ModelBoxRefre(self):
        pt_list = os.listdir('./weights')
        pt_list = [file for file in pt_list if file.endswith(
            '.pt') or file.endswith('.engine')]
        pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x))
        # 必须排序后再比较，否则列表会一直刷新
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            widgets.model_box.clear()
            widgets.model_box.addItems(self.pt_list)

    # 检测置信度参数设置
    def change_val(self, x, flag):
        if flag == 'conf_spinbox':
            widgets.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            widgets.conf_spinbox.setValue(x/100)
            self.show_status('置信度: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100

    # 区域计数
    def region_counter(self):
        if not self.yolo_predict.stop_dtc:
            if self.yolo_predict.region_counter:
                self.yolo_predict.region_counter = False
            else:
                self.yolo_predict.crossing_line = False
                self.yolo_predict.region_counter = True

    # 越线计数
    def crossing_line(self):
        if not self.yolo_predict.stop_dtc:
            if self.yolo_predict.crossing_line:
                self.yolo_predict.crossing_line = False
            else:
                self.yolo_predict.region_counter = False
                self.yolo_predict.crossing_line = True

    # 绘制热力图
    def heatmap(self):
        if not self.yolo_predict.stop_dtc:
            if self.yolo_predict.show_hot_img:
                widgets.second_video.clear()  # 清空视频显示
                # 设置 res_video 占满整个 splitter
                self.set_res_video_fullscreen(widgets)
                self.yolo_predict.show_hot_img = False
            else:
                # 设置 均分 splitter
                self.set_splitter_evenly(widgets)
                self.yolo_predict.show_distence_img = False
                self.yolo_predict.show_speed_img = False
                self.yolo_predict.show_hot_img = True

    # 速度估计
    def speed_estimate(self):
        if not self.yolo_predict.stop_dtc:
            if self.yolo_predict.show_speed_img:
                widgets.second_video.clear()  # 清空视频显示
                # 设置 res_video 占满整个 splitter
                self.set_res_video_fullscreen(widgets)
                self.yolo_predict.show_speed_img = False
            else:
                # 设置 均分 splitter
                self.set_splitter_evenly(widgets)
                self.yolo_predict.show_hot_img = False
                self.yolo_predict.show_distence_img = False
                self.yolo_predict.show_speed_img = True

    # 距离估计
    def distence_estimate(self):
        if not self.yolo_predict.stop_dtc:
            if self.yolo_predict.show_distence_img:
                widgets.second_video.clear()
                # 设置 res_video 占满整个 splitter
                self.set_res_video_fullscreen(widgets)
                self.yolo_predict.show_distence_img = False
            else:
                # 设置 均分 splitter
                self.set_splitter_evenly(widgets)
                self.yolo_predict.show_speed_img = False
                self.yolo_predict.show_hot_img = False
                self.yolo_predict.show_distence_img = True

    # 设置 res_video 占满整个 splitter
    def set_res_video_fullscreen(self, widgets):
        for i in range(widgets.splitter.count()):
            widgets.splitter.setStretchFactor(
                i, 0 if i != widgets.splitter.indexOf(widgets.res_video) else 1)

    # 设置 均分 splitter
    def set_splitter_evenly(self, widgets):
        for i in range(widgets.splitter.count()):
            widgets.splitter.setStretchFactor(
                i, 1 if i in [widgets.splitter.indexOf(widgets.res_video), widgets.splitter.indexOf(widgets.second_video)] else 0)

    # 重置splitter
    def reset_splitter(self, widgets):
        # 获取 splitter 中的所有组件
        children = widgets.splitter.children()
        # 找到 res_video 和 second_video 组件并移除
        to_delete = []
        for child in children:
            if child.objectName() in ["res_video", "second_video"]:
                to_delete.append(child)

        for child in to_delete:
            child.setParent(None)
            child.deleteLater()
        # 创建新的 res_video 组件并添加到 splitter 中
        widgets.second_video = ClickableLabel(self)
        widgets.second_video.setObjectName("second_video")
        widgets.splitter.addWidget(widgets.second_video)
        widgets.res_video = DraggableLabel(self)
        widgets.res_video.setObjectName("res_video")
        widgets.splitter.addWidget(widgets.res_video)
        # 设置 res_video 占满整个 splitter
        self.set_res_video_fullscreen(widgets)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())
