import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, QTabWidget, \
    QComboBox, QHBoxLayout, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import librosa.display
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import convolve, median_filter
from torch import nn
import soundfile as sf
import torch.nn.functional as F
import torchaudio
from getdatapath import *
import tempfile
import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QTabWidget,
    QComboBox,
    QHBoxLayout,
    QSlider,
    QLabel  # 添加 QLabel 导入
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import librosa.display
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import convolve, median_filter
from torch import nn
import soundfile as sf
import torch.nn.functional as F
import torchaudio
from getdatapath import *
import tempfile
import subprocess



# 定义 CNN 网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(5, 5), padding=(2, 2))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

fs = 16000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def audio_enhance(path, method, model):
    window = torch.hann_window(512)
    window = window.to(device)
    data = torchaudio.load(path)[0]
    data = data.to(device)
    stft = torch.stft(data, n_fft=512, hop_length=128, window=window, return_complex=True)
    mm = torch.istft(stft, n_fft=512, hop_length=128, window=window)
    magnitude = torch.squeeze(torch.abs(stft), 0)
    phase = torch.squeeze(torch.angle(stft), 0)

    magnitude = magnitude.to(device)
    phase = phase.to(device)
    feature = torch.unsqueeze(torch.unsqueeze(magnitude.T, 0), 0)
    with torch.no_grad():
        mask = model(feature)
    mask = mask.squeeze(0).squeeze(0).T

    if method == 'direct':
        en_magnitude = mask
    else:
        if method == 'ibm':
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        mask = mask.clamp(min=1e-7, max=1)
        en_magnitude = magnitude * mask

    m=en_magnitude * torch.exp(1j * phase)
    frame = torch.istft(en_magnitude * torch.exp(1j * phase), n_fft=512, hop_length=128, window=window)
    frame = frame.cpu()
    frame = frame.numpy()
    print(frame.shape)
    return frame

# 定义主类 AudioVisualizer
class AudioVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_path = None  # 存储音频文件路径
        self.audio_data = None  # 存储音频数据
        self.sr = None  # 采样率
        self.noisy_spectrogram = None  # 噪声频谱
        self.cleaned_audio = None  # 降噪后的音频
        self.player_original = QMediaPlayer()  # 原始音频播放器
        self.player_denoised = QMediaPlayer()  # 降噪后音频播放器
        self.initUI()  # 初始化用户界面
        self.algorithm = None
        self.setup_hidden_button()  # Add this line to initialize the hidden button

    def setup_hidden_button(self):
        # 创建透明的隐藏按钮
        self.hidden_button = QLabel(self)
        # 设置在右下角，离边缘20像素，大小为15x15像素
        self.hidden_button.setGeometry(self.width() - 35, self.height() - 35, 15, 15)
        # 设置为浅灰色，半透明
        self.hidden_button.setStyleSheet("""
            QLabel {
                background-color: rgba(200, 200, 200, 0.3);
                border: 1px solid rgba(150, 150, 150, 0.5);
                border-radius: 7px;
            }
            QLabel:hover {
                background-color: rgba(200, 200, 200, 0.5);
            }
        """)
        self.hidden_button.mousePressEvent = self.run_changshi

    def run_changshi(self, event):
        try:
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            changshi_path = os.path.join(current_dir, 'changshi.py')

            if os.path.exists(changshi_path):
                # Use subprocess to run the Python script
                subprocess.Popen([sys.executable, changshi_path],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                print("changshi.py not found")
        except Exception as e:
            print(f"Error running changshi.py: {e}")

    def resizeEvent(self, event):
        # Update hidden button position when window is resized
        super().resizeEvent(event)
        if hasattr(self, 'hidden_button'):
            self.hidden_button.setGeometry(self.width() - 10, self.height() - 10, 10, 10)

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('音频降噪工具')
        self.setGeometry(100, 100, 1200, 600)

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 上传按钮
        upload_button = QPushButton('上传音频', self)
        upload_button.clicked.connect(self.load_audio)  # 点击上传按钮触发 load_audio 函数
        layout.addWidget(upload_button)

        # 创建选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_plot_widget(), "音频信号")  # 音频信号选项卡
        self.ml_tab = QWidget()
        self.tab_widget.addTab(self.ml_tab, "机器学习降噪")  # 机器学习降噪选项卡
        self.setup_ml_tab()  # 初始化机器学习选项卡

        self.traditional_tab = QWidget()
        self.tab_widget.addTab(self.traditional_tab, "传统降噪算法")  # 传统降噪算法选项卡
        self.setup_traditional_tab()  # 初始化传统降噪选项卡

        layout.addWidget(self.tab_widget)

        # 初始化定时器，用于更新播放器进度条
        self.timer_original = QTimer()
        self.timer_original.timeout.connect(self.update_slider_original)
        self.timer_denoised = QTimer()
        self.timer_denoised.timeout.connect(self.update_slider_denoised)

    # 创建绘图部件，用于显示音频信号
    def create_plot_widget(self):
        plot_widget = QWidget()
        layout = QHBoxLayout(plot_widget)

        # 原始音频部分
        left_layout = QVBoxLayout()
        layout.addLayout(left_layout)
        self.fig_original = Figure(figsize=(6, 4))  # 创建 Matplotlib 图形对象
        self.canvas_original = FigureCanvas(self.fig_original)  # 将 Matplotlib 图嵌入 PyQt5 窗口
        left_layout.addWidget(self.canvas_original)
        self.play_original_button = QPushButton('播放原始音频', self)
        self.play_original_button.clicked.connect(self.play_original_audio)  # 点击播放原始音频
        left_layout.addWidget(self.play_original_button)
        self.slider_original = QSlider(Qt.Horizontal, self)  # 原始音频滑块
        self.slider_original.setMinimum(0)
        self.slider_original.setMaximum(100)
        self.slider_original.setValue(0)
        self.slider_original.sliderMoved.connect(self.set_position_original)
        left_layout.addWidget(self.slider_original)

        # 降噪音频部分
        right_layout = QVBoxLayout()
        layout.addLayout(right_layout)
        self.fig_denoised = Figure(figsize=(6, 4))  # 创建降噪音频的 Matplotlib 图
        self.canvas_denoised = FigureCanvas(self.fig_denoised)
        right_layout.addWidget(self.canvas_denoised)
        self.play_denoised_button = QPushButton('播放降噪后音频', self)
        self.play_denoised_button.clicked.connect(self.play_denoised_audio)  # 点击播放降噪音频
        right_layout.addWidget(self.play_denoised_button)
        self.slider_denoised = QSlider(Qt.Horizontal, self)  # 降噪音频滑块
        self.slider_denoised.setMinimum(0)
        self.slider_denoised.setMaximum(100)
        self.slider_denoised.setValue(0)
        self.slider_denoised.sliderMoved.connect(self.set_position_denoised)
        right_layout.addWidget(self.slider_denoised)

        return plot_widget

    # 设置机器学习降噪选项卡
    def setup_ml_tab(self):
        layout = QVBoxLayout(self.ml_tab)
        self.ml_algorithm_combo = QComboBox(self)  # 算法选择下拉菜单
        self.ml_algorithm_combo.addItem("请选择算法")
        self.ml_algorithm_combo.addItem("IBM")
        self.ml_algorithm_combo.addItem("IAM")
        self.ml_algorithm_combo.addItem("IRM")
        self.ml_algorithm_combo.addItem("PSM")
        self.ml_algorithm_combo.addItem("ORM")
        self.ml_algorithm_combo.currentIndexChanged.connect(self.apply_ml_denoising_algorithm)
        layout.addWidget(self.ml_algorithm_combo)

    # 设置传统降噪算法选项卡
    def setup_traditional_tab(self):
        layout = QVBoxLayout(self.traditional_tab)
        self.traditional_algorithm_combo = QComboBox(self)  # 传统降噪算法选择菜单
        self.traditional_algorithm_combo.addItem("请选择算法")
        self.traditional_algorithm_combo.addItem("均值滤波器 (Mean Filter)")
        self.traditional_algorithm_combo.addItem("中值滤波器 (Median Filter)")
        self.traditional_algorithm_combo.addItem("维纳滤波器 (Wiener Filter)")
        self.traditional_algorithm_combo.addItem("谱减法 (Spectral Subtraction)")
        self.traditional_algorithm_combo.currentIndexChanged.connect(self.apply_traditional_denoising_algorithm)
        layout.addWidget(self.traditional_algorithm_combo)

    # 加载音频文件
    def load_audio(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "",
                                                   "Audio Files (*.wav *.flac);;All Files (*)", options=options)
        if file_name:
            self.audio_path = file_name  # 保存音频文件路径
            self.plot_audio()  # 绘制音频波形和频谱

    # 绘制原始音频的波形和频谱
    def plot_audio(self):
        try:
            self.audio_data, self.sr = librosa.load(self.audio_path, sr=None)
        except Exception as e:
            print(f"无法加载音频文件: {e}")
            return

        self.fig_original.clear()  # 清空之前的绘图
        ax1 = self.fig_original.add_subplot(211)
        ax2 = self.fig_original.add_subplot(212)
        ax1.plot(self.audio_data)  # 绘制波形
        ax1.set_title('Original Time Domain Waveform')
        ax1.set_xlabel('Sample Points')
        ax1.set_ylabel('Amplitude')
        self.noisy_spectrogram = np.abs(librosa.stft(self.audio_data))  # 计算频谱
        librosa.display.specshow(librosa.amplitude_to_db(self.noisy_spectrogram, ref=np.max), sr=self.sr, x_axis='time',
                                 y_axis='log', cmap='viridis', ax=ax2)
        ax2.set_title('Original Spectrogram')
        ax2.label_outer()
        # 调整子图之间的间距
        self.fig_original.tight_layout()
        self.canvas_original.draw()

    # 应用传统降噪算法
    def apply_traditional_denoising_algorithm(self):
        algorithm = self.traditional_algorithm_combo.currentText()
        if algorithm != "请选择算法":
            if self.audio_data is None:
                print("请先加载音频文件")
                return
            try:
                if algorithm == "均值滤波器 (Mean Filter)":
                    self.cleaned_audio = self.mean_filter(self.audio_data)
                    self.algorithm='Mean Filter'
                elif algorithm == "中值滤波器 (Median Filter)":
                    self.cleaned_audio = self.median_filter(self.audio_data)
                    self.algorithm = 'Median Filter'
                elif algorithm == "维纳滤波器 (Wiener Filter)":
                    self.cleaned_audio = self.wiener_filter(self.audio_data)
                    self.algorithm = 'Wiener Filter'
                elif algorithm == "谱减法 (Spectral Subtraction)":
                    self.cleaned_audio = self.spectral_subtraction(self.audio_data)
                    self.algorithm = 'Spectral Subtraction'
                self.plot_denoised_audio()  # 绘制降噪后的波形和频谱
            except Exception as e:
                print(f"降噪算法出错: {e}")

    # 均值滤波算法
    def mean_filter(self, audio_data, window_size=5):
        kernel = np.ones(window_size) / window_size
        return np.convolve(audio_data, kernel, mode='same')

    # 中值滤波算法
    def median_filter(self, audio_data, window_size=5):
        return median_filter(audio_data, size=window_size)

    # 维纳滤波算法
    def wiener_filter(self, audio_data, noise_floor=0.1):
        noisy_power = np.abs(audio_data) ** 2
        noise_estimate = noise_floor * np.max(noisy_power)
        gain = np.maximum((noisy_power - noise_estimate) / noisy_power, 0)
        return gain * audio_data

    # 谱减法
    def spectral_subtraction(self, audio_data, noise_floor=0.1):
        noise_estimate = noise_floor * np.max(np.abs(audio_data))
        return np.maximum(np.abs(audio_data) - noise_estimate, 0)

    # 应用深度学习降噪算法
    def apply_ml_denoising_algorithm(self):
        algorithm = self.ml_algorithm_combo.currentText()
        if algorithm != "请选择算法":
            if self.audio_data is None:
                print("请先加载音频文件")
                return
            try:
                if algorithm == "IBM":
                    self.cleaned_audio = self.IBM(self.audio_data)
                    self.algorithm = 'IBM'
                elif algorithm == "IAM":
                    self.cleaned_audio = self.IAM(self.audio_data)
                    self.algorithm = 'IAM'
                elif algorithm == "IRM":
                    self.cleaned_audio = self.IRM(self.audio_data)
                    self.algorithm = 'IRM'
                elif algorithm == "ORM":
                    self.cleaned_audio = self.ORM(self.audio_data)
                    self.algorithm = 'ORM'
                elif algorithm == "PSM":
                    self.cleaned_audio = self.PSM(self.audio_data)
                    self.algorithm = 'PSM'
                self.plot_denoised_audio()  # 绘制降噪后的波形和频谱
            except Exception as e:
                print(f"降噪算法出错: {e}")

    def IBM(self,audio_data):
        model = CNN()
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model_' + 'ibm' + '.pth'))
        model.eval()
        noisy_path = self.audio_path
        audio_enhanced = audio_enhance(noisy_path, method = 'ibm',model=model)
        audio_clean = librosa.load(noisy_path, sr=None)[0]
        if len(audio_enhanced) < len(audio_clean):
            pad_length = len(audio_clean) - len(audio_enhanced)
            audio_enhanced = np.pad(audio_enhanced, (0, pad_length), 'constant')
        return audio_enhanced

    def IAM(self,audio_data):
        model = CNN()
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model_' + 'iam' + '.pth'))
        model.eval()
        noisy_path = self.audio_path
        audio_enhanced = audio_enhance(noisy_path, method = 'iam',model=model)
        audio_clean = librosa.load(noisy_path, sr=None)[0]
        if len(audio_enhanced) < len(audio_clean):
            pad_length = len(audio_clean) - len(audio_enhanced)
            audio_enhanced = np.pad(audio_enhanced, (0, pad_length), 'constant')
        return audio_enhanced

    def IRM(self,audio_data):
        model = CNN()
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model_' + 'irm' + '.pth'))
        model.eval()
        noisy_path = self.audio_path
        audio_enhanced = audio_enhance(noisy_path, method = 'irm',model=model)
        audio_clean = librosa.load(noisy_path, sr=None)[0]
        if len(audio_enhanced) < len(audio_clean):
            pad_length = len(audio_clean) - len(audio_enhanced)
            audio_enhanced = np.pad(audio_enhanced, (0, pad_length), 'constant')
        return audio_enhanced

    def ORM(self,audio_data):
        model = CNN()
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model_' + 'orm' + '.pth'))
        model.eval()
        noisy_path = self.audio_path
        audio_enhanced = audio_enhance(noisy_path, method = 'orm',model=model)
        audio_clean = librosa.load(noisy_path, sr=None)[0]
        if len(audio_enhanced) < len(audio_clean):
            pad_length = len(audio_clean) - len(audio_enhanced)
            audio_enhanced = np.pad(audio_enhanced, (0, pad_length), 'constant')
        return audio_enhanced

    def PSM(self,audio_data):
        model = CNN()
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model_' + 'psm' + '.pth'))
        model.eval()
        noisy_path = self.audio_path
        audio_enhanced = audio_enhance(noisy_path, method = 'psm',model=model)
        audio_clean = librosa.load(noisy_path, sr=None)[0]
        if len(audio_enhanced) < len(audio_clean):
            pad_length = len(audio_clean) - len(audio_enhanced)
            audio_enhanced = np.pad(audio_enhanced, (0, pad_length), 'constant')
        return audio_enhanced

    # 绘制降噪后的波形和频谱
    def plot_denoised_audio(self):
        if self.cleaned_audio is None:
            return
        self.fig_denoised.clear()
        ax1 = self.fig_denoised.add_subplot(211)
        ax2 = self.fig_denoised.add_subplot(212)
        ax1.plot(self.cleaned_audio)  # 绘制波形
        ax1.set_title('Denoised Time Domain Waveform')
        ax1.set_xlabel('Sample Points')
        ax1.set_ylabel('Amplitude')
        denoised_spectrogram = np.abs(librosa.stft(self.cleaned_audio))  # 计算频谱
        librosa.display.specshow(librosa.amplitude_to_db(denoised_spectrogram, ref=np.max), sr=self.sr, x_axis='time',
                                 y_axis='log', cmap='viridis', ax=ax2)
        ax2.set_title('Denoised Spectrogram')

        ax1.set_title(f'Denoised Time Domain Waveform ({self.algorithm})')
        ax2.set_title(f'Denoised Spectrogram ({self.algorithm})')

        ax2.label_outer()
        # 调整子图之间的间距
        self.fig_denoised.tight_layout()
        self.canvas_denoised.draw()

    # 播放原始音频
    def play_original_audio(self):
        if self.audio_path:
            url = QUrl.fromLocalFile(self.audio_path)
            content = QMediaContent(url)
            self.player_original.setMedia(content)
            self.player_original.play()
            self.timer_original.start(100)


    def play_denoised_audio(self):
        if self.cleaned_audio is not None:
            try:
                # 使用 `tempfile` 生成唯一的临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file_path = temp_file.name
                    sf.write(temp_file_path, self.cleaned_audio, self.sr)
                    print(f"临时文件保存完成: {temp_file_path}")

                # 播放音频
                url = QUrl.fromLocalFile(temp_file_path)
                content = QMediaContent(url)
                self.player_denoised.setMedia(content)
                self.player_denoised.play()
                self.timer_denoised.start(100)

                # 删除临时文件
                def remove_temp_file(status):
                    if status == QMediaPlayer.EndOfMedia or status == QMediaPlayer.StoppedState:
                        if os.path.exists(temp_file_path):
                            try:
                                os.remove(temp_file_path)
                                print(f"临时文件已删除: {temp_file_path}")
                            except Exception as e:
                                print(f"删除临时文件失败: {e}")

                self.player_denoised.mediaStatusChanged.connect(remove_temp_file)

            except Exception as e:
                print(f"播放降噪音频失败: {e}")

    # 更新原始音频滑块
    def update_slider_original(self):
        if self.player_original.duration() > 0:
            position = int(self.player_original.position() / self.player_original.duration() * 100)
            self.slider_original.setValue(position)
        if self.player_original.state() == QMediaPlayer.StoppedState:
            self.timer_original.stop()

    # 更新降噪音频滑块
    def update_slider_denoised(self):
        if self.player_denoised.duration() > 0:
            position = int(self.player_denoised.position() / self.player_denoised.duration() * 100)
            self.slider_denoised.setValue(position)
        if self.player_denoised.state() == QMediaPlayer.StoppedState:
            self.timer_denoised.stop()

    # 设置原始音频滑块位置
    def set_position_original(self, position):
        duration = self.player_original.duration()
        self.player_original.setPosition(duration * position / 100)

    # 设置降噪音频滑块位置
    def set_position_denoised(self, position):
        duration = self.player_denoised.duration()
        self.player_denoised.setPosition(duration * position / 100)


# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioVisualizer()  # 创建主窗口
    ex.show()  # 显示窗口
    sys.exit(app.exec_())  # 启动应用
