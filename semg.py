import tkinter as tk
from tkinter import messagebox
import serial
import serial.tools.list_ports
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import datetime

class EMGGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG 2通道 訊號收集工具")
        
        # --- 設定視窗為 100% 螢幕大小 ---
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # --- 變數初始化 ---
        self.serial_port = None
        self.ser = None
        self.baud_rate = 921600
        self.is_recording = False
        self.thread_running = True
        
        self.data_queue = queue.Queue()
        self.all_data_log = []
        
        # 設定為 2 個通道
        self.num_channels = 2
        self.y_data = [[] for _ in range(self.num_channels)]
        
        self.current_label = ""
        self.current_filename = ""

        # --- 建立 GUI 介面 ---
        self.setup_ui()
        self.setup_plot()
        
        # --- 啟動背景執行緒與更新迴圈 ---
        self.serial_thread = threading.Thread(target=self.read_serial_task)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        self.update_plot()
        
        # 處理視窗關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """ 建立上方控制面板 """
        control_frame = tk.Frame(self.root, pady=10, padx=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(control_frame, text="發音文字 (Label):", font=("Arial", 14)).pack(side=tk.LEFT, padx=(0, 5))
        self.entry_label = tk.Entry(control_frame, font=("Arial", 14), width=15)
        self.entry_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(control_frame, text="▶ 開始錄製", font=("Arial", 14, "bold"), width=12, command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(control_frame, text="■ 停止並存檔", font=("Arial", 14, "bold"), width=12, state=tk.DISABLED, command=self.stop_recording)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("狀態: 待機中，請輸入發音文字並按下開始。")
        self.lbl_status = tk.Label(control_frame, textvariable=self.status_var, font=("Arial", 12))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

    def setup_plot(self):
        """ 建立 Matplotlib 圖表並嵌入 Tkinter """
        plt.rcParams['figure.dpi'] = 100 
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        
        # 設定 2 條線的顏色 (綠色為 Ch1, 黃色為 Ch2)
        colors = ['#00ff00', '#ffff00']
        self.lines = []
        for i in range(self.num_channels):
            line, = self.ax.plot([], [], lw=1.5, color=colors[i], label=f'Ch{i+1}')
            self.lines.append(line)
        
        # 樣式設定
        self.ax.set_facecolor('black') 
        self.fig.patch.set_facecolor('#222222') 
        self.ax.set_ylim(0, 4200)   
        self.ax.set_xlim(0, 500)    
        self.ax.set_title("Real-time 2-Channel sEMG Signal", color='white')
        self.ax.set_xlabel("Time (Samples)", color='white')
        self.ax.set_ylabel("ADC Value", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.grid(True, color='#444444', linestyle='--')
        self.ax.legend(loc='upper right', facecolor='black', labelcolor='white')

        # 將圖表嵌入 Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def find_mac_port(self):
        """ 嘗試自動尋找 STM32 的 Port """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'usbmodem' in port.device:
                return port.device
        return '/dev/tty.usbmodem1103' # 預設值

    def connect_serial(self):
        """ 連接 Serial Port """
        try:
            port_name = self.find_mac_port()
            self.ser = serial.Serial(port_name, self.baud_rate, timeout=1)
            self.ser.reset_input_buffer()
            return True
        except Exception as e:
            messagebox.showerror("連線失敗", f"無法連接 Serial Port。\n錯誤訊息: {e}\n\n請檢查硬體是否插好。")
            return False

    def start_recording(self):
        """ 開始錄製按鈕事件 """
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
        
        # 清空 2 個通道的圖表與暫存
        for i in range(self.num_channels):
            self.y_data[i].clear()
            self.lines[i].set_data([], [])
            
        while not self.data_queue.empty():
            self.data_queue.get()
            
        self.canvas.draw_idle()

        self.ax.set_title(f"Real-time 2-Channel sEMG - Label: {self.current_label}", color='white')
        self.entry_label.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set(f"狀態: 🔴 錄製中... (標籤: {self.current_label})")
        
        if self.ser:
            self.ser.reset_input_buffer()
        self.is_recording = True

    def stop_recording(self):
        """ 停止錄製按鈕事件 """
        self.is_recording = False
        
        if self.all_data_log:
            try:
                with open(self.current_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'ch1', 'ch2', 'label'])
                    writer.writerows(self.all_data_log)
                messagebox.showinfo("存檔成功", f"成功儲存 {len(self.all_data_log)} 筆數據！\n\n檔名: {self.current_filename}")
            except Exception as e:
                messagebox.showerror("存檔失敗", f"存檔時發生錯誤: {e}")
        else:
            messagebox.showwarning("警告", "沒有收集到任何數據，未產生檔案。")

        self.entry_label.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("狀態: 🟢 已停止，準備好進行下一次錄製。")
        self.ax.set_title("Real-time 2-Channel sEMG Signal", color='white')
        self.canvas.draw_idle()

    def read_serial_task(self):
        """ 背景執行緒：負責讀取 Serial 數據 """
        while self.thread_running:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        raw_line = self.ser.readline()
                        print(raw_line)
                        
                        if self.is_recording:
                            line = raw_line.decode('utf-8', errors='ignore').strip()
                            parts = line.split(',')
                            
                            # 預期收到 2 個數據 (例如 "2048,1500")
                            if len(parts) == self.num_channels:
                                try:
                                    vals = [int(p) for p in parts]
                                    print('recive:', vals)
                                    timestamp = time.time()
                                    self.data_queue.put(vals)
                                    # 將 [時間, ch1, ch2, 標籤] 存入 log
                                    self.all_data_log.append([timestamp] + vals + [self.current_label])
                                except ValueError:
                                    pass # 忽略雜訊
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    print(f"讀取錯誤: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def update_plot(self):
        """ 更新圖表 """
        if self.is_recording:
            updated = False
            while not self.data_queue.empty():
                vals = self.data_queue.get()
                for i in range(self.num_channels):
                    self.y_data[i].append(vals[i])
                    if len(self.y_data[i]) > 500:
                        self.y_data[i].pop(0)
                updated = True
            
            if updated:
                x_data = range(len(self.y_data[0]))
                for i in range(self.num_channels):
                    self.lines[i].set_data(x_data, self.y_data[i])
                self.ax.set_xlim(0, max(500, len(self.y_data[0])))
                self.canvas.draw_idle()  
        
        if self.thread_running:
            self.root.after(20, self.update_plot)

    def on_closing(self):
        """ 視窗關閉時的清理動作 """
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