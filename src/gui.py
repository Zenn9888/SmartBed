from tkinter import *
from tkinter import ttk, Frame, Button, Label
import numpy as np
import pandas as pd
from tkinter.font import Font
import time
from model import PatientModel  # Import the PatientModel
from tkinter import filedialog  # 添加這行在文件頂部的import區域


class PatientGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("智慧簽床系統")
        self.master.geometry("1250x1000")

        # 將視窗置中
        self.center_window()

        # 定義新的配色方案 - 藍白主題
        self.colors = {
            "bg_white": "#ffffff",  # 主背景白
            "bg_light": "#f5f6fa",  # 淺灰背景
            "primary": "#2962ff",  # 主要藍色
            "secondary": "#0d47a1",  # 深藍色
            "accent": "#40c4ff",  # 亮藍色
            "text_dark": "#2c3e50",  # 主要文字色
            "text_light": "#ffffff",  # 白色文字
            "hover": "#1e88e5",  # 懸停效果色
            "border": "#e3e3e3",  # 邊框色
        }

        self.master.configure(bg=self.colors["bg_white"])

        # 分頁屬性
        self.bed_current_page = 0
        self.bed_page_size = 24
        self.current_page = 0
        self.page_size = 10

        # 載入資料
        self.bed_data = pd.read_csv("病房病床編號.csv")
        self.ward_list = sorted(self.bed_data["HNURSTAT"].unique())

        # 添加按鈕框架
        self.create_model_buttons()

        # 初始化模型
        self.model = PatientModel()

        # 創建主要容器
        self.create_header()
        self.create_main_container()
        self.create_ward_selector()
        self.create_bed_display()
        self.create_stats_panel()

        # 設置初始病房
        if self.ward_list:
            self.ward_dropdown.set(self.ward_list[0])
            self.update_bed_buttons(None)

    def center_window(self):
        # 獲取螢幕寬度和高度
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # 計算視窗的x和y座標
        x = (screen_width - 1250) // 2
        y = (screen_height - 900) // 2

        # 設置視窗位置
        self.master.geometry(f"1150x850+{x}+{y}")

    def create_header(self):
        header = Frame(self.master, bg=self.colors["bg_white"], height=100)
        header.pack(fill=X, pady=(20, 0))

        # 創建現代感標題
        title_frame = Frame(header, bg=self.colors["bg_white"])
        title_frame.pack(expand=True)

        title_font = Font(family="Helvetica", size=32, weight="bold")
        title = Label(
            title_frame,
            text="智慧簽床系統",
            bg=self.colors["bg_white"],
            fg=self.colors["primary"],
            font=title_font,
        )
        title.pack(pady=20)

        # 添加子標題
        subtitle = Label(
            title_frame,
            text="Advanced Patient Distribution System",
            bg=self.colors["bg_white"],
            fg=self.colors["text_dark"],
            font=("Helvetica", 14),
        )
        subtitle.pack()

    def create_ward_selector(self):
        selector_frame = Frame(self.master, bg=self.colors["bg_light"])
        selector_frame.pack(fill=X, pady=20, padx=40)

        # 添加圓角效果的選擇器背景
        selector_inner = Frame(
            selector_frame, bg=self.colors["bg_white"], padx=20, pady=15
        )
        selector_inner.pack(fill=X)

        Label(
            selector_inner,
            text="選擇病房",
            bg=self.colors["bg_white"],
            fg=self.colors["text_dark"],
            font=("Helvetica", 14, "bold"),
        ).pack(side=LEFT, padx=20)

        # 自定義下拉選單樣式
        style = ttk.Style()
        style.configure(
            "Custom.TCombobox",
            background=self.colors["bg_white"],
            fieldbackground=self.colors["bg_white"],
            foreground=self.colors["text_dark"],
        )

        self.selected_ward = StringVar()
        self.ward_dropdown = ttk.Combobox(
            selector_inner,
            textvariable=self.selected_ward,
            values=self.ward_list,
            state="readonly",
            style="Custom.TCombobox",
            font=("Helvetica", 12),
            width=20,
        )
        self.ward_dropdown.pack(side=LEFT, padx=10)
        self.ward_dropdown.bind("<<ComboboxSelected>>", self.update_bed_buttons)

    def create_main_container(self):
        self.main_container = Frame(self.master, bg=self.colors["bg_white"])
        self.main_container.pack(expand=True, fill=BOTH, padx=20, pady=20)

    def create_bed_display(self):
        # 左側面板
        self.bed_panel = Frame(self.main_container, bg=self.colors["bg_light"])
        self.bed_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        # 床位按鈕容器
        self.bed_buttons_frame = Frame(self.bed_panel, bg=self.colors["bg_white"])
        self.bed_buttons_frame.pack(expand=True, fill=BOTH, padx=20, pady=20)

        # 分頁導航 - 現代風格
        nav_frame = Frame(self.bed_panel, bg=self.colors["bg_white"])
        nav_frame.pack(fill=X, pady=10)

        self.bed_prev_button = Button(
            nav_frame,
            text="◀",
            command=self.prev_bed_page,
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 12),
            bd=0,
            width=4,
            cursor="hand2",
            relief="flat",
        )
        self.bed_prev_button.pack(side=LEFT, padx=5)

        self.bed_page_label = Label(
            nav_frame,
            text="第1頁",
            bg=self.colors["bg_white"],
            fg=self.colors["text_dark"],
            font=("Helvetica", 12),
        )
        self.bed_page_label.pack(side=LEFT, padx=20)

        self.bed_next_button = Button(
            nav_frame,
            text="▶",
            command=self.next_bed_page,
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 12),
            bd=0,
            width=4,
            cursor="hand2",
            relief="flat",
        )
        self.bed_next_button.pack(side=LEFT, padx=5)

    def create_stats_panel(self):
        # 創建統計面板框架
        self.stats_frame = Frame(self.main_container, bg=self.colors["bg_white"])
        self.stats_frame.pack(side=RIGHT, fill=Y, padx=20)

        # 標題
        title = Label(
            self.stats_frame,
            text="推薦病人列表",
            bg=self.colors["bg_white"],
            fg=self.colors["primary"],
            font=("Helvetica", 16, "bold"),
        )
        title.pack(pady=10)

        # 結果顯示區域
        self.results_content = Label(
            self.stats_frame,
            text="選擇床位查看推薦列表",
            bg=self.colors["bg_light"],
            fg=self.colors["text_dark"],
            font=("Helvetica", 12),
            width=30,
            height=20,
            wraplength=250,
        )
        self.results_content.pack(pady=10)

        # 分頁控制框架
        nav_frame = Frame(self.stats_frame, bg=self.colors["bg_white"])
        nav_frame.pack(pady=10)

        # 上一頁按鈕
        self.prev_button = Button(
            nav_frame,
            text="上一頁",
            command=self.prev_page,
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 10),
            bd=0,
            padx=10,
            pady=5,
            cursor="hand2",
            relief="flat",
            state="disabled",
        )
        self.prev_button.pack(side=LEFT, padx=5)

        # 頁碼標籤
        self.page_label = Label(
            nav_frame,
            text="第1頁",
            bg=self.colors["bg_white"],
            fg=self.colors["text_dark"],
            font=("Helvetica", 10),
        )
        self.page_label.pack(side=LEFT, padx=10)

        # 下一頁按鈕
        self.next_button = Button(
            nav_frame,
            text="下一頁",
            command=self.next_page,
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 10),
            bd=0,
            padx=10,
            pady=5,
            cursor="hand2",
            relief="flat",
            state="disabled",
        )
        self.next_button.pack(side=LEFT, padx=5)

    def create_model_buttons(self):
        # 創建按鈕框架
        button_frame = Frame(self.master, bg=self.colors["bg_white"])
        button_frame.pack(pady=10)

        # 載入訓練資料按鈕
        load_train_button = Button(
            button_frame,
            text="載入訓練資料",
            command=lambda: self.load_patient_data(is_training=True),
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 10),
            bd=0,
            padx=15,
            pady=5,
            cursor="hand2",
            relief="flat",
        )
        load_train_button.pack(side=LEFT, padx=5)

        # 載入預測資料按鈕
        load_pred_button = Button(
            button_frame,
            text="載入預測資料",
            command=lambda: self.load_patient_data(is_training=False),
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 10),
            bd=0,
            padx=15,
            pady=5,
            cursor="hand2",
            relief="flat",
        )
        load_pred_button.pack(side=LEFT, padx=5)

        # 訓練按鈕
        train_button = Button(
            button_frame,
            text="訓練模型",
            command=self.train_model,
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 10),
            bd=0,
            padx=15,
            pady=5,
            cursor="hand2",
            relief="flat",
        )
        train_button.pack(side=LEFT, padx=5)

        # 載入模型按鈕
        load_button = Button(
            button_frame,
            text="載入模型",
            command=self.load_model,
            bg=self.colors["primary"],
            fg=self.colors["text_light"],
            font=("Helvetica", 10),
            bd=0,
            padx=15,
            pady=5,
            cursor="hand2",
            relief="flat",
        )
        load_button.pack(side=LEFT, padx=5)

    def load_patient_data(self, is_training=True):
        try:
            # 顯示檔案選擇對話框
            file_path = filedialog.askopenfilename(
                title="選擇" + ("訓練" if is_training else "預測") + "資料CSV檔案",
                filetypes=[("CSV files", "*.csv")],
                initialdir=".",
            )

            if not file_path:  # 如果使用者取消選擇
                self.results_content.config(text="已取消載入資料")
                return

            # 顯示載入中的訊息
            self.results_content.config(text="載入資料中... 請稍候...")
            self.master.update()

            # 載入資料
            success, message = self.model.load_patient_data(file_path, is_training)

            if success:
                if is_training:
                    self.results_content.config(text=f"{message}\n可以開始訓練模型了！")
                else:
                    self.results_content.config(text=f"{message}\n可以開始進行預測了！")
            else:
                self.results_content.config(text=message)

        except Exception as e:
            self.results_content.config(text=f"載入資料時發生錯誤：\n{str(e)}")

    def train_model(self):
        try:
            # 檢查是否已載入資料
            if (
                not hasattr(self.model, "training_data")
                or self.model.training_data is None
            ):
                self.results_content.config(text="請先載入病人資料！")
                return

            # 顯示訓練中的訊息
            self.results_content.config(text="訓練模型中... 請稍後...")
            self.master.update()

            # 訓練模型
            train_score, test_score = self.model.train("1131218簽床紀錄.csv")

            # 顯示訓練結果
            result_text = (
                f"模型訓練完畢！\n"
                f"訓練準確度: {train_score:.2%}\n"
                f"測試準確度: {test_score:.2%}"
            )
            self.results_content.config(text=result_text)

        except Exception as e:
            self.results_content.config(text=f"Error during training:\n{str(e)}")

    def load_model(self):
        try:
            # 顯示載入中的訊息
            self.results_content.config(text="請選擇模型檔案...")
            self.master.update()

            # 讓使用者選擇模型檔案
            model_file = filedialog.askopenfilename(
                title="選擇模型檔案",
                filetypes=[("Joblib files", "*.joblib")],
                initialdir="model",
            )

            if not model_file:  # 如果使用者取消選擇
                self.results_content.config(text="已取消載入模型")
                return

            # 顯示載入中的訊息
            self.results_content.config(text="載入模型中... 請稍候...")
            self.master.update()

            # 載入模型
            success = self.model.load_model(model_file)

            if success:
                self.results_content.config(text="模型載入成功！")
            else:
                self.results_content.config(
                    text="載入模型失敗。\n請確認檔案格式是否正確。"
                )

        except Exception as e:
            self.results_content.config(text=f"載入模型時發生錯誤：\n{str(e)}")

    def update_bed_buttons(self, event):
        for widget in self.bed_buttons_frame.winfo_children():
            widget.destroy()

        ward = self.selected_ward.get()
        self.ward_beds = self.bed_data[self.bed_data["HNURSTAT"] == ward]
        self.bed_current_page = 0
        self.display_bed_page()

    def display_bed_page(self):
        for widget in self.bed_buttons_frame.winfo_children():
            widget.destroy()

        start = self.bed_current_page * self.bed_page_size
        end = start + self.bed_page_size
        page_beds = self.ward_beds.iloc[start:end]

        # 創建科技感床位按鈕
        for i, row in enumerate(page_beds.iterrows()):
            bed_id = row[1]["HBEDNO"]

            # 創建按鈕容器來實現特殊效果
            btn_frame = Frame(
                self.bed_buttons_frame, bg=self.colors["primary"], padx=2, pady=2
            )

            button = Button(
                btn_frame,
                text=f"{bed_id}",
                command=lambda id=bed_id,
                ward=self.selected_ward.get(): self.show_patient_probabilities(
                    id, ward
                ),
                bg=self.colors["bg_white"],
                fg=self.colors["primary"],
                font=("Helvetica", 12, "bold"),
                width=10,
                height=2,
                bd=0,
                cursor="hand2",
                relief="flat",
            )
            button.pack(padx=1, pady=1)

            row_num = i // 6
            col_num = i % 6
            btn_frame.grid(row=row_num, column=col_num, padx=8, pady=8, sticky="nsew")

            # 改進的懸停效果
            def on_enter(e, btn=button, frame=btn_frame):
                btn.config(bg=self.colors["accent"], fg=self.colors["text_light"])
                frame.config(bg=self.colors["secondary"])

            def on_leave(e, btn=button, frame=btn_frame):
                btn.config(bg=self.colors["bg_white"], fg=self.colors["primary"])
                frame.config(bg=self.colors["primary"])

            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)

        # 更新頁碼顯示
        max_pages = -(-len(self.ward_beds) // self.bed_page_size)
        self.bed_page_label.config(
            text=f"第 {self.bed_current_page + 1} 頁，共 {max_pages} 頁"
        )

        # 更新導航按鈕狀態
        self.bed_prev_button.config(
            state="normal" if self.bed_current_page > 0 else "disabled"
        )
        self.bed_next_button.config(
            state="normal" if self.bed_current_page < max_pages - 1 else "disabled"
        )

    def show_patient_probabilities(self, bed_id, ward):
        self.results_content.config(text="計算中...")
        self.master.update()

        try:
            # 檢查模型是否已載入
            if (
                not hasattr(self.model, "is_model_loaded")
                or not self.model.is_model_loaded
            ):
                self.results_content.config(text="請先載入模型！")
                return

            # 獲取床位索引
            bed_index = self.bed_data[
                (self.bed_data["HNURSTAT"] == ward)
                & (self.bed_data["HBEDNO"] == bed_id)
            ].index[0]

            # 獲取該床位的所有病人機率
            probabilities = self.model.get_bed_probabilities(bed_index)

            if probabilities is None:
                self.results_content.config(text="無法計算機率，請確認模型狀態")
                return

            # 獲取病人ID（優先使用MASKED_HHISNUM，如果沒有則使用HHISNUM）
            if "MASKED_HHISNUM" in self.model.prediction_data.columns:
                patient_ids = self.model.prediction_data["MASKED_HHISNUM"].values
            elif "HHISNUM" in self.model.prediction_data.columns:
                patient_ids = self.model.prediction_data["HHISNUM"].values
            else:
                self.results_content.config(text="預測資料中缺少病人ID欄位")
                return

            # 創建病人ID和機率的配對
            all_patients_probs = list(zip(patient_ids, probabilities))

            # 排序結果
            self.all_patients_probs = sorted(
                all_patients_probs, key=lambda x: x[1], reverse=True
            )
            self.current_page = 0
            self.display_page()

        except Exception as e:
            self.results_content.config(text=f"處理錯誤：\n{str(e)}")

    def display_page(self):
        if not hasattr(self, "all_patients_probs") or not self.all_patients_probs:
            self.results_content.config(text="無資料顯示")
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            self.page_label.config(text="Page 0")
            return

        start = self.current_page * self.page_size
        end = start + self.page_size
        page_data = self.all_patients_probs[start:end]

        max_pages = -(-len(self.all_patients_probs) // self.page_size)  # 向上取整

        result_text = f"床位機率分析結果\n"
        result_text += f"第 {self.current_page + 1} 頁，共 {max_pages} 頁\n"
        result_text += "=" * 40 + "\n\n"

        for patient_id, prob in page_data:
            result_text += f"病歷號: {patient_id} 機率: {prob:.2%}\n"

        self.results_content.config(text=result_text)
        self.page_label.config(text=f"第 {self.current_page + 1} 頁，共 {max_pages} 頁")

        # 更新按鈕狀態
        self.prev_button.config(state="normal" if self.current_page > 0 else "disabled")
        self.next_button.config(
            state="normal" if self.current_page < max_pages - 1 else "disabled"
        )

    def next_page(self):
        if hasattr(self, "all_patients_probs"):
            max_pages = -(-len(self.all_patients_probs) // self.page_size)
            if self.current_page < max_pages - 1:
                self.current_page += 1
                self.display_page()

    def prev_page(self):
        if hasattr(self, "all_patients_probs"):
            if self.current_page > 0:
                self.current_page -= 1
            self.display_page()

    def next_bed_page(self):
        max_pages = -(-len(self.ward_beds) // self.bed_page_size)
        if self.bed_current_page < max_pages - 1:
            self.bed_current_page += 1
            self.display_bed_page()

    def prev_bed_page(self):
        if self.bed_current_page > 0:
            self.bed_current_page -= 1
            self.display_bed_page()


if __name__ == "__main__":
    root = Tk()
    root.title("Neural Patient Interface")
    app = PatientGUI(root)
    root.mainloop()
