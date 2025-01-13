import sys
import numpy as np
import cv2
import librosa
import soundfile as sf
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


class SpectrogramConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语谱图转音频工具")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 创建左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # 添加输入字段
        self.time_end_input = QLineEdit()
        self.freq_end_input = QLineEdit()
        control_layout.addWidget(QLabel("结束时间 (秒):"))
        control_layout.addWidget(self.time_end_input)
        control_layout.addWidget(QLabel("结束频率 (Hz):"))
        control_layout.addWidget(self.freq_end_input)

        # 添加上传按钮
        self.upload_btn = QPushButton("上传图像")
        self.upload_btn.clicked.connect(self.upload_image)
        control_layout.addWidget(self.upload_btn)

        # 添加转换按钮
        self.convert_btn = QPushButton("转换为音频")
        self.convert_btn.clicked.connect(self.convert_to_audio)
        control_layout.addWidget(self.convert_btn)

        # 添加播放控制
        self.play_btn = QPushButton("播放音频")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)

        # 添加停止按钮
        self.stop_btn = QPushButton("停止播放")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        # 添加进度标签
        self.progress_label = QLabel("")
        control_layout.addWidget(self.progress_label)

        control_layout.addStretch()
        layout.addWidget(control_panel)

        # 创建右侧图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        layout.addWidget(self.image_label)

        # 初始化媒体播放器
        self.player = QMediaPlayer()
        self.player.stateChanged.connect(self.media_state_changed)

        # 存储当前图像路径
        self.current_image_path = None
        self.current_audio_path = None

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择语谱图图像", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.current_image_path = file_name
            # 显示图像
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

            # 清除之前的音频路径
            self.current_audio_path = None
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def convert_to_audio(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先上传图像！")
            return

        try:
            time_end = float(self.time_end_input.text())
            freq_end = float(self.freq_end_input.text())

            self.progress_label.setText("正在处理图像...")
            QApplication.processEvents()

            # 加载和处理图像
            img, linear_spectrogram, _, hop_length, n_fft, sr = self.load_spectrogram_image(
                self.current_image_path, time_end, freq_end
            )

            self.progress_label.setText("正在转换为音频...")
            QApplication.processEvents()

            # 转换为音频
            window_length = n_fft // 2
            audio = self.custom_griffin_lim(
                linear_spectrogram,
                n_iter=64,
                hop_length=hop_length,
                win_length=window_length,
                n_fft=n_fft
            )

            # 增强音频
            audio_enhanced = self.enhance_audio(audio, sr)

            # 保存音频
            self.current_audio_path = 'reconstructed_audio_enhanced.wav'
            sf.write(self.current_audio_path, audio_enhanced, sr)

            # 启用播放按钮
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

            self.progress_label.setText("转换完成！")
            QMessageBox.information(self, "成功", "音频转换完成！")

        except Exception as e:
            self.progress_label.setText("转换失败")
            QMessageBox.critical(self, "错误", f"转换过程中出现错误：{str(e)}")

    def play_audio(self):
        if self.current_audio_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.current_audio_path)))
            self.player.setVolume(50)
            self.player.play()

    def stop_audio(self):
        self.player.stop()

    def media_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        elif state == QMediaPlayer.PlayingState:
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    # 以下是从原始代码复制的处理函数
    def load_spectrogram_image(self, image_path, time_end, freq_end, time_start=0, freq_start=0, min_dB=-80.0,
                               max_dB=0.0):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("无法加载图像，请检查路径是否正确。")

        img_height, img_width = img.shape
        img_flipped = np.flipud(img)
        log_spectrogram = max_dB - (img_flipped / 255.0) * (max_dB - min_dB)
        linear_spectrogram = np.power(10.0, log_spectrogram / 20.0, dtype=np.float32)
        linear_spectrogram = signal.medfilt2d(linear_spectrogram, kernel_size=3)

        duration = time_end - time_start
        sr = int(2.2 * freq_end)
        hop_length = max(int((sr * duration) / img_width), 1)
        n_fft = 2 * (img_height - 1)

        return img, linear_spectrogram, log_spectrogram, hop_length, n_fft, sr

    def normalize_spectrogram(self, S, percentile=99):
        S_norm = S.copy()
        threshold = np.percentile(S_norm, percentile)
        S_norm = np.clip(S_norm, 0, threshold)
        S_max = np.max(np.abs(S_norm)) if np.max(np.abs(S_norm)) > 0 else 1
        S_norm = S_norm / S_max
        return S_norm

    def apply_phase_reconstruction(self, S, n_iter=64, hop_length=None, win_length=None, n_fft=None):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S = S.astype(np.complex64, copy=False)
        momentum = np.zeros_like(angles)

        for i in range(n_iter):
            angles_with_momentum = angles + 0.9 * momentum
            S_complex = S * angles_with_momentum / np.abs(angles_with_momentum)
            inverse = librosa.istft(S_complex, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
            rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            angles_new = rebuilt / np.abs(rebuilt)
            momentum = 0.9 * momentum + 0.1 * (angles_new - angles)
            angles = angles_new

            # 更新进度
            self.progress_label.setText(f"相位重建进度: {i + 1}/{n_iter}")
            QApplication.processEvents()

        return S * angles

    def custom_griffin_lim(self, S, n_iter=64, hop_length=None, win_length=None, n_fft=None, window='hann'):
        if win_length is None:
            win_length = n_fft // 2

        if hop_length is None:
            hop_length = win_length // 4

        window = signal.windows.kaiser(win_length, beta=8)
        fft_window = librosa.filters.get_window(window, win_length).astype(np.float32)

        S = self.normalize_spectrogram(S, percentile=99)
        complex_spec = self.apply_phase_reconstruction(S, n_iter, hop_length, win_length, n_fft)

        inverse = librosa.istft(complex_spec,
                                hop_length=hop_length,
                                win_length=win_length,
                                window=fft_window)

        inverse = signal.wiener(inverse, mysize=5)
        inverse = librosa.util.normalize(inverse)

        return inverse

    def enhance_audio(self, audio, sr):
        audio_enhanced = signal.lfilter([1, -0.97], [1], audio)
        threshold = 0.3
        ratio = 0.6
        mask = np.abs(audio_enhanced) > threshold
        audio_enhanced[mask] = threshold + (np.abs(audio_enhanced[mask]) - threshold) * ratio * np.sign(
            audio_enhanced[mask])
        return audio_enhanced


def main():
    app = QApplication(sys.argv)
    window = SpectrogramConverter()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()