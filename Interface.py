import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import os
import importlib
import backend


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

CROP_WIDTH, CROP_HEIGHT = 450, 120
PREVIEW_WIDTH, PREVIEW_HEIGHT = 450, 300
IMG_WIDTH, IMG_HEIGHT = 1000, 700

class PlateReaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.minsize(1000, 700)
        self.title("سامانه پردازش پلاک")
        self.geometry("1600x880")
        self.icon_path = "icon.ico"
        if os.path.exists(self.icon_path):
            self.iconbitmap(self.icon_path)

        self.tk_img = None
        self.img_original = None

        self.setup_ui()

    def setup_ui(self):
        self.add_credits()
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.create_header()
        self.create_left_panel()
        self.create_right_panel()

    def create_header(self):
        header = ctk.CTkLabel(self, text="📷 سامانه هوشمند خوانش پلاک خودرو", font=("B Nazanin", 32, "bold"), anchor="center", justify="center")
        header.grid(row=0, column=0, columnspan=2, pady=20, sticky="new")

    def create_left_panel(self):
        left_frame = ctk.CTkFrame(self, corner_radius=15)
        left_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=0)
        left_frame.grid_columnconfigure(0, weight=1)

        self.canvas_input = ctk.CTkCanvas(left_frame, width=IMG_WIDTH, height=IMG_HEIGHT, bg="black", highlightthickness=0)
        self.canvas_input.grid(row=0, column=0, padx=20, pady=20)

        self.load_btn = ctk.CTkButton(left_frame, text="📁 بارگذاری تصویر",
                                 font=("B Nazanin", 18, "bold"), command=self.load_image)
        self.load_btn.grid(row=1, column=0, pady=(0, 10), sticky="s")

    def create_right_panel(self):
        right_frame = ctk.CTkFrame(self, corner_radius=15)
        right_frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")

        ctk.CTkLabel(right_frame, text="برش پلاک", font=("B Nazanin", 16, "bold"), anchor="e", justify="right").pack(pady=(20, 10))
        self.canvas_crop = ctk.CTkCanvas(right_frame, width=CROP_WIDTH, height=CROP_HEIGHT,
                                         bg="white", highlightthickness=0)
        self.canvas_crop.pack(pady=5)

        ctk.CTkLabel(right_frame, text=":پلاک تشخیص داده شده", font=("B Nazanin", 16), anchor="e", justify="right").pack(pady=(20, 5))
        self.text_output = ctk.CTkLabel(right_frame, text=" ", font=("B Nazanin", 28, "bold"),
                                        text_color="#66ff99", anchor="e", justify="right")
        self.text_output.pack(pady=(5, 20))

        ctk.CTkLabel(right_frame, text="🕘 آخرین تصویر", font=("B Nazanin", 16, "bold"), anchor="e", justify="right").pack(pady=(10, 5))
        self.canvas_previous = ctk.CTkCanvas(right_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT,
                                             bg="#1e1e2f", highlightthickness=0)
        self.canvas_previous.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("تصاویر", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return

        self.original_image = Image.open(path)
        img = self.original_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        self.img_original = img
        enhancer = ImageEnhance.Brightness(img)
        self.fade_step = 0
        self.fade_frames = [ImageTk.PhotoImage(enhancer.enhance(i / 10.0)) for i in range(1, 11)]

        def fade_in():
            if self.fade_step < len(self.fade_frames):
                self.canvas_input.delete("all")
                self.canvas_input.create_image(0, 0, anchor="nw", image=self.fade_frames[self.fade_step])
                self.tk_img = self.fade_frames[self.fade_step]
                self.fade_step += 1
                self.after(50, fade_in)
            else:
                self.animate_scan()

        fade_in()

    def animate_scan(self):
        self.scan_y = 0
        self.canvas_input.delete("scan")
        self.scan_line = self.canvas_input.create_line(0, 0, IMG_WIDTH, 0, fill="", width=2, tags="scan")
        self.scan()

    def scan(self):
        self.canvas_input.delete("scan")
        color = f"#{hex(0x66 + (self.scan_y % 50))[2:]}ff99"
        self.canvas_input.create_line(0, self.scan_y, IMG_WIDTH, self.scan_y,
                                      fill=color, width=2, tags="scan")
        self.scan_y += 8
        if self.scan_y < IMG_HEIGHT:
            self.after(10, self.scan)
        else:
            self.canvas_input.delete("scan")
            self.after(200, self.process_plate)  # ← تغییر در این خط

    def process_plate(self):
        try:
            importlib.reload(backend)
            result, plate_crop = backend.predict_plate(self.img_original)
            self.animate_text(self.text_output, result)

            plate_crop = plate_crop.resize((CROP_WIDTH, CROP_HEIGHT))
            self.tk_crop = ImageTk.PhotoImage(plate_crop)
            self.canvas_crop.delete("all")
            self.canvas_crop.create_image(0, 0, anchor="nw", image=self.tk_crop)

        except Exception as e:
            self.animate_text(self.text_output, f"خطا: {e}")

        self.show_previous()

    def animate_text(self, label, text):
        label.configure(text="")
        def type_char(i=0):
            if i <= len(text):
                label.configure(text=text[:i])
                self.after(100, lambda: type_char(i + 1))
        type_char()

    def add_credits(self):
        credit = ctk.CTkLabel(self, text="© 2025 Sina Eslami. All rights reserved.",
                             font=("Times New Roman", 12), text_color="#888", anchor="center", justify="center")
        credit.grid(row=2, column=0, columnspan=2, pady=(5, 10), sticky="s")

    def show_previous(self):
        previous = self.img_original.copy().resize((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Brightness(previous)
        self.fade_prev_step = 0
        self.fade_prev_frames = [ImageTk.PhotoImage(enhancer.enhance(i / 10.0)) for i in range(1, 11)]

        def fade_in_prev():
            if self.fade_prev_step < len(self.fade_prev_frames):
                self.canvas_previous.delete("all")
                self.canvas_previous.create_image(0, 0, anchor="nw", image=self.fade_prev_frames[self.fade_prev_step])
                self.tk_previous = self.fade_prev_frames[self.fade_prev_step]
                self.fade_prev_step += 1
                self.after(50, fade_in_prev)

        fade_in_prev()


if __name__ == "__main__":
    app = PlateReaderApp()
    app.mainloop()
