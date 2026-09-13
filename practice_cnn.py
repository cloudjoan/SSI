import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import serial
import serial.tools.list_ports
import time
import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import datetime
import joblib 

# 匯入 TensorFlow 與 Keras 相關套件 (讀取 CNN 大腦必備)
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 解決 Matplotlib 中文顯示問題 ---
plt.rcParams['font.sans-serif'] = ['PingFang TC', 'Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# 關閉 TF 煩人的提示訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EMGGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG 終極訊號工具 (示波器/對齊/CNN 深度學習即時預測)")
        
        # --- 設定視窗為 100% 螢幕大小 ---
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        self.serial_port = None
        self.ser = None
        self.baud_rate = 115200 
        self.is_recording = False
        self.thread_running = True
        
        self.data_queue = queue.Queue()
        self.all_data_log = []
        
        # 設定為 2 個通道
        self.num_channels = 2
        self.y_data = [[] for _ in range(self.num_channels)]
        self.ch_vars = [] 
        
        self.current_label = ""
        self.current_filename = ""

        # --- CNN AI 預測相關設定 ---
        self.window_size = 250 # 必須與訓練時的 WINDOW_SIZE 一致 (0.5秒)
        self.model_loaded = False
        self.cnn_model = None  # 存放 TensorFlow CNN 模型
        self.scaler = None     # 存放特徵縮放器
        self.le = None         # 存放標籤解碼器
        self.last_predict_time = time.time() # 控制預測頻率避免卡頓

        self.setup_ui()
        self.setup_plot()
        
        self.serial_thread = threading.Thread(target=self.read_serial_task)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        self.update_plot()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(500, self.connect_serial)

    def scan_available_models(self):
        """ 掃描 pkl 目錄下的所有 emg_cnn_model_*.h5，萃取出肌肉名稱 """
        # 改為尋找 CNN 的 .h5 檔案
        model_files = glob.glob(os.path.join("pkl", "emg_cnn_model_*.h5"))
        muscle_names = []
        for file in model_files:
            filename = os.path.basename(file)
            # 檔名格式: emg_cnn_model_zgm_abd.h5 -> 替換掉前後綴
            name = filename.replace("emg_cnn_model_", "").replace(".h5", "")
            muscle_names.append(name)
        return muscle_names

    def load_ai_model(self, event=None):
        """ 根據下拉選單載入 CNN 模型 (.h5) 與配件 (.pkl) """
        selected_muscle = self.cb_model.get()
        if not selected_muscle or selected_muscle == "無可用模型":
            return

        model_path = os.path.join("pkl", f"emg_cnn_model_{selected_muscle}.h5")
        scaler_path = os.path.join("pkl", f"emg_scaler_{selected_muscle}.pkl")
        encoder_path = os.path.join("pkl", f"emg_label_encoder_{selected_muscle}.pkl")

        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
                # 載入 TensorFlow Keras 模型
                self.cnn_model = load_model(model_path)
                # 載入 Scikit-learn 配件
                self.scaler = joblib.load(scaler_path)
                self.le = joblib.load(encoder_path)
                
                # 進行一次假預測 (Warm-up)，讓 GPU 暖機，避免第一次預測時卡頓
                dummy_input = np.zeros((1, self.window_size, 2))
                self.cnn_model.predict(dummy_input, verbose=0)
                
                self.model_loaded = True
                self.lbl_predict.config(text=f"CNN AI 預測 ({selected_muscle}): 等待訊號...", bg="yellow", fg="blue")
                print(f"🧠 成功載入【{selected_muscle}】專屬 CNN 大腦與硬體加速！")
            else:
                self.model_loaded = False
                self.lbl_predict.config(text="⚠️ 找不到完整的模型配件", bg="red", fg="white")
                messagebox.showerror("模型載入失敗", f"找不到 {selected_muscle} 的完整 .h5 與 .pkl 檔案！")
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("讀取錯誤", f"載入 CNN 模型時發生錯誤: {e}")

    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=5, padx=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        row1_frame = tk.Frame(control_frame)
        row1_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        row2_frame = tk.Frame(control_frame)
        row2_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(row1_frame, text="發音文字 (Label):", font=("Arial", 12)).pack(side=tk.LEFT, padx=(0, 5))
        self.entry_label = tk.Entry(row1_frame, font=("Arial", 12), width=15)
        self.entry_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(row1_frame, text="▶ 開始錄製", font=("Arial", 12, "bold"), width=12, command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(row1_frame, text="■ 停止並存檔", font=("Arial", 12, "bold"), width=12, state=tk.DISABLED, command=self.stop_recording)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.ch_frame = tk.Frame(row1_frame)
        self.ch_frame.pack(side=tk.LEFT, padx=15)
        tk.Label(self.ch_frame, text="顯示通道:", font=("Arial", 12)).pack(side=tk.LEFT)
        for i in range(self.num_channels):
            var = tk.BooleanVar(value=True)
            self.ch_vars.append(var)
            chk = tk.Checkbutton(self.ch_frame, text=f"Ch{i+1}", variable=var, font=("Arial", 12), command=self.toggle_visibility)
            chk.pack(side=tk.LEFT, padx=2)
            
        self.status_var = tk.StringVar()
        self.status_var.set("狀態: 待機中，尋找硬體設備...")
        self.lbl_status = tk.Label(row1_frame, textvariable=self.status_var, font=("Arial", 12))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        tk.Label(row2_frame, text="🧠 選擇 CNN 模型 (肌肉部位):", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        available_muscles = self.scan_available_models()
        self.cb_model = ttk.Combobox(row2_frame, values=available_muscles, state="readonly", font=("Arial", 12), width=15)
        self.cb_model.pack(side=tk.LEFT, padx=5)
        self.cb_model.bind("<<ComboboxSelected>>", self.load_ai_model)
        
        self.lbl_predict = tk.Label(row2_frame, text="CNN 預測: (請先選擇模型)", font=("Arial", 18, "bold"), fg="gray", bg="#eeeeee", padx=15, pady=2)
        self.lbl_predict.pack(side=tk.LEFT, padx=30)

        self.shift_frame = tk.Frame(row2_frame)
        self.shift_frame.pack(side=tk.RIGHT, padx=15)
        tk.Label(self.shift_frame, text="視圖對齊 (Samples):", font=("Arial", 12)).pack(side=tk.LEFT)
        self.shift_var = tk.IntVar(value=0)
        self.slider_shift = tk.Scale(self.shift_frame, variable=self.shift_var, from_=-50, to=50, orient=tk.HORIZONTAL, length=150, showvalue=True)
        self.slider_shift.pack(side=tk.LEFT, padx=5)

        if available_muscles:
            self.cb_model.current(0) 
            self.load_ai_model()     
        else:
            self.cb_model.set("無可用模型")

    def setup_plot(self):
        plt.rcParams['figure.dpi'] = 100 
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        colors = ['#00ff00', '#ffff00']
        self.lines = []
        self.value_texts = [] 
        
        for i in range(self.num_channels):
            line, = self.ax.plot([], [], lw=1.5, color=colors[i], label=f'Ch{i+1}')
            self.lines.append(line)
            txt = self.ax.text(0.02, 0.95 - (i * 0.08), f"Ch{i+1}: 0", 
                               transform=self.ax.transAxes, color=colors[i], 
                               fontsize=14, fontweight='bold',
                               bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
            self.value_texts.append(txt)
        
        self.ax.set_facecolor('black') 
        self.fig.patch.set_facecolor('#222222') 
        self.ax.set_ylim(0, 4200)   
        self.ax.set_xlim(0, 500)    
        self.ax.set_title("Real-time Oscilloscope & CNN Prediction", color='white')
        self.ax.set_xlabel("Time (Samples)", color='white')
        self.ax.set_ylabel("ADC Value", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.grid(True, color='#444444', linestyle='--')
        self.ax.legend(loc='upper right', facecolor='black', labelcolor='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def toggle_visibility(self):
        for i in range(self.num_channels):
            if i < len(self.lines):
                is_visible = self.ch_vars[i].get()
                self.lines[i].set_visible(is_visible)
                if i < len(self.value_texts):
                    self.value_texts[i].set_visible(is_visible)
        self.canvas.draw_idle()

    def find_mac_port(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'usbmodem' in port.device:
                return port.device
        return '/dev/tty.usbmodem1103' 

    def connect_serial(self):
        try:
            port_name = self.find_mac_port()
            self.ser = serial.Serial(port_name, self.baud_rate, timeout=1)
            self.ser.reset_input_buffer()
            self.status_var.set(f"狀態: 🟢 已連線 ({port_name})")
            return True
        except Exception as e:
            self.status_var.set("狀態: 🔴 連線失敗，請檢查硬體。")
            return False

    def start_recording(self):
        label_text = self.entry_label.get().strip()
        if not label_text:
            messagebox.showwarning("警告", "請先輸入「發音文字」！")
            return
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial():
                return

        self.current_label = label_text
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        safe_label = "".join(c for c in label_text if c.isalnum() or c in (' ', '_', '-')).rstrip()
        self.current_filename = f"{safe_label}_{timestamp_str}_emg_2ch_data.csv"
        
        self.all_data_log.clear()
        for i in range(self.num_channels):
            self.y_data[i].clear()
            self.lines[i].set_data([], [])
            
        while not self.data_queue.empty():
            self.data_queue.get()
            
        self.canvas.draw_idle()
        self.ax.set_title(f"[RECORDING] Label: {self.current_label}", color='#ff4444')
        self.entry_label.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set(f"狀態: 🔴 錄製中... (標籤: {self.current_label})")
        
        if self.ser:
            self.ser.reset_input_buffer()
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        if self.all_data_log:
            try:
                with open(self.current_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'ch1', 'ch2', 'label'])
                    writer.writerows(self.all_data_log)
                messagebox.showinfo("存檔成功", f"成功儲存數據！\n\n檔名: {self.current_filename}")
            except Exception as e:
                messagebox.showerror("存檔失敗", f"存檔時發生錯誤: {e}")

        self.entry_label.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("狀態: 🟢 已停止，恢復預覽模式。")
        self.ax.set_title("Real-time Oscilloscope & CNN Prediction", color='white')
        self.canvas.draw_idle()

    def read_serial_task(self):
        while self.thread_running:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        raw_line = self.ser.readline()
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                        parts = line.split(',')
                        if len(parts) == self.num_channels:
                            try:
                                vals = [int(p) for p in parts]
                                timestamp = time.time()
                                self.data_queue.put(vals)
                                if self.is_recording:
                                    self.all_data_log.append([timestamp] + vals + [self.current_label])
                            except ValueError:
                                pass 
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def update_plot(self):
        updated = False
        while not self.data_queue.empty():
            vals = self.data_queue.get()
            for i in range(self.num_channels):
                self.y_data[i].append(vals[i])
                if len(self.y_data[i]) > 600:
                    self.y_data[i].pop(0)
            updated = True
        
        if updated and len(self.y_data[0]) > 0:
            shift = self.shift_var.get()
            n_len = min(len(self.y_data[0]), len(self.y_data[1]))
            plot_y = [[], []]
            
            if n_len > abs(shift) + 1:
                if shift > 0: 
                    plot_y[0] = self.y_data[0][:n_len-shift]
                    plot_y[1] = self.y_data[1][shift:n_len]
                elif shift < 0:
                    shift_abs = abs(shift)
                    plot_y[0] = self.y_data[0][shift_abs:n_len]
                    plot_y[1] = self.y_data[1][:n_len-shift_abs]
                else:
                    plot_y[0] = self.y_data[0][:n_len]
                    plot_y[1] = self.y_data[1][:n_len]
            else:
                plot_y[0] = self.y_data[0][:n_len]
                plot_y[1] = self.y_data[1][:n_len]

            if len(plot_y[0]) > 0:
                display_y0 = plot_y[0][-500:]
                display_y1 = plot_y[1][-500:]
                x_data = range(len(display_y0))
                self.lines[0].set_data(x_data, display_y0)
                self.value_texts[0].set_text(f"Ch1: {display_y0[-1]}")
                self.lines[1].set_data(x_data, display_y1)
                self.value_texts[1].set_text(f"Ch2: {display_y1[-1]}")
                self.ax.set_xlim(0, 500)
                self.canvas.draw_idle()  
            
            # 🚀 核心：CNN 深度學習即時推論 (限流機制：每 0.2 秒最多預測一次，保護 UI 不卡頓)
            current_time = time.time()
            if self.model_loaded and len(self.y_data[0]) >= self.window_size and (current_time - self.last_predict_time > 0.2):
                
                # 永遠抓最新的 250 個點做預測
                win_ch1 = self.y_data[0][-self.window_size:]
                win_ch2 = self.y_data[1][-self.window_size:]
                
                try:
                    # 1. 將兩條通道合併成 (250, 2) 的二維陣列 (無須特徵萃取，保留原始波形結構！)
                    window_raw = np.column_stack((win_ch1, win_ch2))
                    
                    # 2. 依照訓練時的 StandardScaler 進行縮放
                    # 由於 scaler 當初是針對 (N, 2) 訓練的，這裡直接對 (250, 2) transform
                    window_scaled = self.scaler.transform(window_raw)
                    
                    # 3. 為了符合 CNN 的輸入格式 (Batch_Size, Time_Steps, Channels)
                    # 我們把 (250, 2) 增加一個維度變成 (1, 250, 2)
                    X_input = np.expand_dims(window_scaled, axis=0)
                    
                    # 4. CNN 預測 (回傳的是各類別的機率矩陣)
                    pred_probs = self.cnn_model.predict(X_input, verbose=0)
                    
                    # 5. 找出機率最高的那一個類別代碼
                    pred_code = np.argmax(pred_probs, axis=1)
                    
                    # 6. 透過標籤解碼器把數字轉回文字 ('a', 'æ', '0'...)
                    pred_text = self.le.inverse_transform(pred_code)[0]
                    confidence = np.max(pred_probs) * 100 # 計算信心水準
                    
                    # 顯示在畫面上
                    self.lbl_predict.config(text=f"CNN: 【 {pred_text} 】 (信心: {confidence:.0f}%)", bg="#e3f2fd", fg="#1565c0")
                    self.last_predict_time = current_time # 更新預測時間
                except Exception as e:
                    print(f"CNN 預測錯誤: {e}")
                    pass 
        
        if self.thread_running:
            # 提高 UI 更新頻率，讓示波器更滑順
            self.root.after(15, self.update_plot)

    def on_closing(self):
        self.is_recording = False
        self.thread_running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()
        print("🔌 程式已安全關閉。")

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGGUIApp(root)
    root.mainloop()