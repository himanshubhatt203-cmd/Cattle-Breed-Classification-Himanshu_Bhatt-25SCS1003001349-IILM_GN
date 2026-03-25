import os
import json
import threading
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path

# This import must point to where your model class is defined.
# If 'src.model' is not a module, you will need to adjust this.
# Example: If your model class is in a file named 'cattle_model.py' in the same directory,
# you would use: from cattle_model import CattleBreedClassifier
# !!! The model file (e.g., 'src/model.py') must be accessible and contain the class 'CattleBreedClassifier'
# !!! If the path is relative, ensure the environment is set up to find it.

import torch
from torchvision import transforms

# === EDIT THESE PATHS ===
# The model and label file paths MUST be updated to your actual locations.
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "best_model.pth"
LABELS_FILE = BASE_DIR / "models" / "labels.json"
MODEL_CLASS_PATH = "src.model"
MODEL_CLASS_NAME = "CattleBreedClassifier"
NUM_CLASSES = 15
INPUT_SIZE = (224, 224)
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load breed labels
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")
with open(LABELS_FILE, "r") as f:
    breed_names = json.load(f)

# === ABOUT INFO FOR EACH BREED ===
# Keys must match the breed names coming from labels.json.
BREED_INFO = {
    "Gir": (
        "Gir (Gyr) is an indigenous dairy breed from Gujarat (Junagadh, Amreli, Bhavnagar regions). "
        "Life span is typically 12–15 years, often more with good care. "
        "Average milk production is about 1,900–2,100 L per lactation (≈5–7 L/day). "
        "Usual feeding includes around 6–10 kg roughage and 1–4 kg concentrate daily. "
        "Common vaccinations: FMD at ~4 months, HS at 6 months, with annual boosters."
    ),
    "Sahiwal": (
        "Sahiwal is a high-yielding milch breed from the Punjab region along the India–Pakistan border. "
        "Life span is about 12–15 years. Milk production ranges from 1,600–2,750 kg per lactation (≈6–10 L/day). "
        "Typical feed is 8–12 kg roughage and 2–6 kg concentrate. "
        "They follow the standard vaccination schedule (FMD, HS, BQ annually)."
    ),
    "Red_Sindhi": (
        "Red Sindhi is a hardy milch breed originating from the Sindh region and also reared in parts of India. "
        "Life span is generally 12+ years. Milk production is around 1,100–2,600 kg per lactation. "
        "They usually consume 6–10 kg roughage and 1–4 kg concentrate daily. "
        "Vaccination follows the standard national schedule (FMD, HS, BQ)."
    ),
    "Tharparkar": (
        "Tharparkar is a dual-purpose, drought-resistant breed from the Thar Desert region of Rajasthan. "
        "Life span is about 12–15 years. Milk production ranges from 800–2,100 kg per lactation "
        "(average ~1,700 kg). Typical feed is 6–10 kg roughage and 1–3 kg concentrate. "
        "They are well adapted to arid and semi-arid conditions and follow the standard annual vaccination schedule."
    ),
    "Rathi": (
        "Rathi is a dual-purpose cattle breed from Rajasthan (Bikaner, Jaisalmer, Ganganagar areas). "
        "Life span is about 12–15 years. Milk production is around 1,000–1,800 kg per lactation "
        "(average ~1,560 kg). It usually needs about 6–9 kg roughage plus a smaller quantity of concentrate. "
        "Regular FMD, HS, and BQ vaccinations are recommended."
    ),
    "Ongole": (
        "Ongole (Nellore) is a famous draught and meat-type breed from Andhra Pradesh (Nellore, Guntur regions). "
        "Life span is typically 12+ years. Milk production varies widely from about 600–2,500 kg per lactation. "
        "They usually consume 8–12 kg roughage and 2–4 kg concentrate daily. "
        "They follow the normal vaccination schedule (FMD, HS, BQ, etc.)."
    ),
    "Kankrej": (
        "Kankrej is a dual-purpose breed from Gujarat and Rajasthan. "
        "Life span is about 12–15 years. Milk production is around 1,700–1,800 kg per lactation. "
        "Typical feed includes 8–12 kg roughage and 2–5 kg concentrate. "
        "They are strong draught animals and follow the standard vaccination schedule."
    ),
    "Murrah": (
        "Murrah is a premier buffalo breed mainly from Haryana, Punjab, and Western Uttar Pradesh. "
        "Life span is around 10–15 years. Milk production is about 1,500–2,000 kg per lactation, "
        "with high-yielding animals reaching 10–20 L/day. "
        "Feed requirement is about 10–15 kg roughage and 4–8 kg concentrate daily. "
        "Vaccinations include FMD, HS, BQ, and often Brucellosis, with annual boosters."
    ),
    "Surti": (
        "Surti is a dairy buffalo breed from Gujarat (Anand, Vadodara, Kaira regions). "
        "Life span is around 10–14 years. Milk yield is approximately 1,500–2,000 L per lactation "
        "(≈4–6 L/day). Usual feed is 8–12 kg roughage plus 2–5 kg concentrate. "
        "They follow regular vaccinations (FMD, HS, BQ)."
    ),
    "Jaffrabadi": (
        "Jaffrabadi is a heavy, high-yield buffalo breed from Gujarat (Jamnagar, Junagadh areas). "
        "Life span is about 10–14 years. Milk production is around 2,000–2,300 kg per lactation "
        "(≈8–15 L/day) with rich, high-fat milk. "
        "Feed requirement is about 10–15 kg roughage and 4–6 kg concentrate daily. "
        "They follow the standard vaccination schedule."
    ),
    "Mehsana": (
        "Mehsana is a dairy buffalo breed developed in the Mehsana region of Gujarat. "
        "Life span is around 10–14 years. Typical milk production is about 1,800–2,000 L per lactation "
        "(≈7–9 L/day). Usual feed intake is 9–14 kg roughage and 3–6 kg concentrate daily. "
        "Vaccinations include FMD, HS, and BQ annually."
    ),
    "Hariana": (
        "Hariana is a dual-purpose (milk and draught) indigenous breed from Haryana "
        "(Rohtak, Hisar, Karnal districts). Life span is about 12–15 years. "
        "Milk yield ranges from 700–1,700 kg per lactation (≈5–10 L/day). "
        "They usually need 8–12 kg roughage plus 1–4 kg concentrate. "
        "FMD, HS, and BQ vaccinations are recommended annually."
    ),
    "Deoni": (
        "Deoni is a dual-purpose breed from Maharashtra and Karnataka. "
        "Life span is typically 12+ years. Milk production is around 600–1,200 kg per lactation "
        "(average ~870 L). It usually requires 6–10 kg roughage and 1–3 kg concentrate daily. "
        "They follow the standard vaccination schedule."
    ),
    "Vechur": (
        "Vechur is a famous dwarf cow breed from Kerala (Vechoor village and nearby areas). "
        "Life span is generally 12+ years. Milk production is about 500–900 kg per lactation "
        "(≈3–5 L/day), with very economical feed requirements. "
        "They need only a few kilograms of green fodder and a small portion of concentrate. "
        "They follow the standard vaccination protocol (FMD, HS, etc.) and are well-suited to humid/tropical climates."
    ),
    "Holstein_Friesian": (
        "Holstein Friesian (HF) is an exotic dairy breed known for very high milk production. "
        "Life span can be 15–20 years, though productive life is often 6–10 years. "
        "Typical milk yield ranges from 8,000–12,000+ L per lactation, and can exceed 20,000 kg "
        "in top-managed farms. Feed demand is high: about 40–60 kg green fodder, "
        "10–20 kg dry fodder, and 8–12 kg concentrate daily. "
        "They require full dairy herd vaccination (FMD, HS, Brucellosis, mastitis control, etc.) "
        "and are best suited to cooler climates or well-managed farms across India."
    ),
}

# You could also add a generic fallback note here:
GENERIC_BREED_NOTE = (
    "\n\nGeneral note: Follow standard vaccination schedule in India "
    "(FMD at ~4 months, HS and BQ around 6 months, Brucellosis in young female calves, "
    "with annual boosters). Balanced feeding with green fodder, dry fodder, concentrates, "
    "minerals, and clean water is essential for good milk yield and health."
)


def import_model_class(module_path, class_name):
    import importlib
    m = importlib.import_module(module_path)
    return getattr(m, class_name)


ModelClass = import_model_class(MODEL_CLASS_PATH, MODEL_CLASS_NAME)

# Preprocessing transform
PIL_TO_TENSOR = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE[0]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class BreedVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breedify")
        self.root.geometry("1400x800")
        self.colors = {
            'background': '#F5F1E8', 'card': '#FFFFFF', 'card_secondary': '#FEFDFB',
            'primary': '#8B5A3C', 'primary_light': '#A87456', 'accent': '#6B8E4E',
            'text': '#3D2817', 'text_secondary': '#6B5A4D', 'text_muted': '#9A8B7D',
            'border': '#E5DDD0', 'border_hover': '#8B5A3C', 'success': '#6B8E4E', 'shadow': '#D4C4B0'
        }
        self.root.configure(bg=self.colors['background'])

        self.model = None
        self.model_loaded_flag = False
        self.selected_image_path = None
        self.is_analyzing = False
        self.history = []
        self.tk_image = None  # keep reference to current Tk image

        self.setup_ui()
        self.show_status("Loading model...")
        threading.Thread(target=self.load_model_thread, daemon=True).start()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['card'], height=90,
                                 highlightbackground=self.colors['border'], highlightthickness=1)
        header_frame.pack(fill=tk.X, pady=(0, 0))
        header_frame.pack_propagate(False)

        header_content = tk.Frame(header_frame, bg=self.colors['card'])
        header_content.pack(pady=20, padx=30, anchor='w')

        title_container = tk.Frame(header_content, bg=self.colors['card'])
        title_container.pack(side=tk.LEFT)

        icon_frame = tk.Frame(title_container, bg=self.colors['primary'], width=50, height=50)
        icon_frame.pack(side=tk.LEFT, padx=(0, 15))
        icon_frame.pack_propagate(False)
        icon_label = tk.Label(icon_frame, text="🌾", font=("Segoe UI Emoji", 20),
                              bg=self.colors['primary'], fg="white")
        icon_label.place(relx=0.5, rely=0.5, anchor='center')

        text_container = tk.Frame(title_container, bg=self.colors['card'])
        text_container.pack(side=tk.LEFT)
        title_label = tk.Label(text_container, text="Breedify", font=("Segoe UI", 20, "bold"),
                               bg=self.colors['card'], fg=self.colors['text'])
        title_label.pack(anchor='w')
        subtitle_label = tk.Label(text_container, text="AI-Powered Cattle Breed Identification",
                                  font=("Segoe UI", 9),
                                  bg=self.colors['card'], fg=self.colors['text_muted'])
        subtitle_label.pack(anchor='w')

        self.status_label = tk.Label(header_frame, text="", bg=self.colors['card'],
                                     fg=self.colors['text_muted'])
        self.status_label.pack(side=tk.RIGHT, padx=20)

        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        left_frame = tk.Frame(main_container, bg=self.colors['background'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # Upload Card
        self.upload_card = self.create_card(left_frame)
        self.upload_card.pack(fill=tk.BOTH, pady=(0, 15))

        upload_header = tk.Label(self.upload_card.inner, text="Upload Image", font=("Segoe UI", 13, "bold"),
                                 bg=self.colors['card'], fg=self.colors['text'], anchor='w')
        upload_header.pack(fill=tk.X, pady=(0, 15))

        self.image_container = tk.Frame(self.upload_card.inner, bg=self.colors['card_secondary'],
                                         highlightbackground=self.colors['border'], highlightthickness=2)
        self.image_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Placeholder label (we will later put the image here)
        self.image_label = tk.Label(self.image_container, text="", bg=self.colors['card_secondary'],
                                     fg=self.colors['text_muted'], font=("Segoe UI", 13, "bold"))
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.upload_icon_label = tk.Label(self.image_label, text="🖼", font=("Segoe UI Emoji", 48),
                                         bg=self.colors['card_secondary'], fg=self.colors['text_muted'])
        self.upload_icon_label.pack(pady=(30, 10))
        self.upload_text_label = tk.Label(
            self.image_label,
            text="Upload Cattle Image\n\nSelect a clear image of the cattle for breed identification",
            font=("Segoe UI", 10), bg=self.colors['card_secondary'],
            fg=self.colors['text_muted'], justify=tk.CENTER
        )
        self.upload_text_label.pack()

        button_container = tk.Frame(self.upload_card.inner, bg=self.colors['card'])
        button_container.pack(pady=15)
        self.button_container = button_container

        self.upload_btn = self.create_button(
            button_container, "📁  Choose Image", self.select_image,
            bg_color=self.colors['primary'], hover_color=self.colors['primary_light']
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = self.create_button(
            button_container, "🔍  Identify Breed", self.analyze_image,
            bg_color=self.colors['accent'], hover_color='#7BA05A', state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        # Results Card
        self.results_card = self.create_card(left_frame)
        self.results_card.pack(fill=tk.BOTH, expand=True)
        results_header = tk.Label(self.results_card.inner, text="Prediction Results",
                                  font=("Segoe UI", 13, "bold"),
                                  bg=self.colors['card'], fg=self.colors['text'], anchor='w')
        results_header.pack(fill=tk.X, pady=(0, 15))

        results_scroll_frame = tk.Frame(self.results_card.inner, bg=self.colors['card'])
        results_scroll_frame.pack(fill=tk.BOTH, expand=True)

        results_scrollbar = tk.Scrollbar(results_scroll_frame)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text = tk.Text(
            results_scroll_frame, font=("Segoe UI", 10), bg=self.colors['card'],
            fg=self.colors['text'], wrap=tk.WORD, height=12, state=tk.DISABLED,
            relief=tk.FLAT, padx=10, pady=10, yscrollcommand=results_scrollbar.set,
            borderwidth=0
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        results_scrollbar.config(command=self.results_text.yview)

        self.upload_again_btn = self.create_button(
            self.results_card.inner, "🔄 Upload Again", self.reset_app,
            bg_color="#555555", hover_color="#777777"
        )
        self.upload_again_btn.pack(fill=tk.X, pady=(15, 0))
        self.upload_again_btn.pack_forget()  # Hide initially

        # Right side history
        history_card = self.create_card(main_container, width=350)
        history_card.pack(side=tk.RIGHT, fill=tk.BOTH)
        history_card.pack_propagate(False)

        history_header_frame = tk.Frame(history_card.inner, bg=self.colors['card'])
        history_header_frame.pack(fill=tk.X, pady=(0, 15))

        history_icon = tk.Label(history_header_frame, text="📜", font=("Segoe UI Emoji", 14),
                                 bg=self.colors['card'])
        history_icon.pack(side=tk.LEFT, padx=(0, 8))
        history_header = tk.Label(history_header_frame, text="Prediction History",
                                  font=("Segoe UI", 13, "bold"),
                                  bg=self.colors['card'], fg=self.colors['text'], anchor='w')
        history_header.pack(side=tk.LEFT)

        self.history_count_label = tk.Label(
            history_header_frame, text="0", font=("Segoe UI", 9, "bold"),
            bg=self.colors['border'], fg=self.colors['text'], padx=8, pady=2
        )
        self.history_count_label.pack(side=tk.RIGHT)

        history_scroll_frame = tk.Frame(history_card.inner, bg=self.colors['card'])
        history_scroll_frame.pack(fill=tk.BOTH, expand=True)

        history_scrollbar = tk.Scrollbar(history_scroll_frame)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_listbox = tk.Listbox(
            history_scroll_frame, font=("Segoe UI", 9), bg=self.colors['card'],
            fg=self.colors['text'], yscrollcommand=history_scrollbar.set,
            selectbackground=self.colors['primary'], selectforeground="white",
            highlightthickness=0, borderwidth=0, relief=tk.FLAT, activestyle='none'
        )
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=5)
        self.history_listbox.bind('<<ListboxSelect>>', self.on_history_select)
        history_scrollbar.config(command=self.history_listbox.yview)

        self.history_empty_label = tk.Label(
            history_scroll_frame,
            text="📋\n\nNo predictions yet\n\nUpload and analyze images\nto build your history",
            font=("Segoe UI", 9), bg=self.colors['card'],
            fg=self.colors['text_muted'], justify=tk.CENTER
        )
        self.history_empty_label.pack(expand=True)

    def create_card(self, parent, width=None):
        card = tk.Frame(parent, bg=self.colors['card'],
                         highlightbackground=self.colors['border'], highlightthickness=1)
        if width:
            card.configure(width=width)
        inner = tk.Frame(card, bg=self.colors['card'])
        inner.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)
        card.inner = inner
        return card

    def create_button(self, parent, text, command, bg_color, hover_color, state=tk.NORMAL):
        btn = tk.Button(
            parent, text=text, command=command, font=("Segoe UI", 11, "bold"),
            bg=bg_color, fg="white", activebackground=hover_color,
            activeforeground="white", padx=25, pady=12,
            cursor="hand2", relief=tk.FLAT, borderwidth=0, state=state
        )

        def on_enter(e):
            if btn['state'] != tk.DISABLED:
                btn['background'] = hover_color

        def on_leave(e):
            if btn['state'] != tk.DISABLED:
                btn['background'] = bg_color

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def show_status(self, text):
        self.status_label.config(text=text)

    def load_model_thread(self):
        try:
            model = ModelClass(num_classes=NUM_CLASSES).to(device)
            state = torch.load(MODEL_FILE, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            else:
                model.load_state_dict(state)
            model.eval()
            self.model = model
            self.model_loaded_flag = True
            self.root.after(0, self.on_model_loaded)
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Model Load Error", f"Failed to load model: {err}"))
            self.root.after(0, lambda: self.show_status("Model load failed"))

    def on_model_loaded(self):
        self.show_status("Model loaded ✔")
        self.analyze_btn.config(state=tk.NORMAL)
        self.root.after(1500, lambda: self.show_status(""))

    def reset_upload_card(self):
        self.upload_card.configure(width=None, height=None)
        self.upload_card.pack_propagate(True)
        self.image_label.configure(text="", image="", font=("Segoe UI", 13, "bold"),
                                   bg=self.colors['card_secondary'], fg=self.colors['primary'])
        self.upload_icon_label.pack(pady=(30, 10))
        self.upload_text_label.pack()
        self.button_container.pack(pady=15)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Cattle Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.selected_image_path = file_path

                # Hide placeholder
                self.upload_icon_label.pack_forget()
                self.upload_text_label.pack_forget()

                # Load and show preview image
                img = Image.open(file_path).convert("RGB")
                img.thumbnail((400, 400))  # keep aspect ratio within 400x400
                self.tk_image = ImageTk.PhotoImage(img)

                self.image_label.configure(
                    image=self.tk_image,
                    text="",
                    bg=self.colors['card_secondary'],
                    fg=self.colors['primary']
                )
                self.image_label.image = self.tk_image  # prevent GC
                self.image_container.configure(highlightbackground=self.colors['border_hover'])

                # Update results text
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Click 'Identify Breed' to analyze the image")
                self.results_text.config(state=tk.DISABLED)

                # Enable analyze if model is ready
                if self.model_loaded_flag:
                    self.analyze_btn.config(state=tk.NORMAL)
                else:
                    self.analyze_btn.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def expand_results_shrink_upload(self):
        # Shrink upload card but keep current image preview
        self.upload_card.configure(width=320, height=180)
        self.upload_card.pack_propagate(False)
        self.image_label.configure(anchor='w', justify='left')
        self.button_container.pack(fill=tk.X)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.results_card.configure(width=950, height=600)
        self.results_card.pack(fill=tk.BOTH, expand=True)
        self.results_card.pack_propagate(True)

    def analyze_image(self):
        if not self.selected_image_path:
            return
        if not self.model_loaded_flag:
            messagebox.showwarning("Model not loaded", "Model is still loading. Please wait.")
            return
        if self.is_analyzing:
            return
        self.is_analyzing = True
        self.analyze_btn.config(state=tk.DISABLED, text="⏳  Analyzing...")
        self.upload_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_analysis_thread, daemon=True).start()

    def run_analysis_thread(self):
        try:
            img = Image.open(self.selected_image_path).convert("RGB")
            x = PIL_TO_TENSOR(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                pred_idx_item = int(pred_idx.item())
                conf_item = float(conf.item()) * 100.0
                key = str(pred_idx_item)
                breed = breed_names.get(key, f"index_{pred_idx_item}")
                topk = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
                top_preds = []
                for idx, p in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
                    top_preds.append(
                        {"breed": breed_names.get(str(idx), f"index_{idx}"),
                         "confidence": int(p * 100)}
                    )
            prediction = {
                "breed": breed,
                "confidence": round(conf_item, 2),
                "top_predictions": top_preds
            }
            self.root.after(0, lambda: self.display_results(prediction))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Prediction Error", f"Inference failed: {e}"))
            self.root.after(0, lambda: self.analysis_cleanup())

    def analysis_cleanup(self):
        self.is_analyzing = False        
        self.analyze_btn.config(state=tk.NORMAL, text="🔍  Identify Breed")
        self.upload_btn.config(state=tk.NORMAL)

    def display_results(self, prediction):
        self.expand_results_shrink_upload()
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "✔ ", "icon")
        self.results_text.insert(tk.END, "Primary Breed Identified\n\n", "section_header")
        self.results_text.insert(tk.END, f"{prediction['breed']}\n", "breed_name")
        self.results_text.insert(
            tk.END,
            f"● {prediction['confidence']}% confident\n\n",
            "confidence"
        )
        self.results_text.insert(tk.END, "─" * 60 + "\n\n", "separator")

        # 👉 About section using BREED_INFO
        self.results_text.insert(tk.END, f"ℹ  About {prediction['breed']}\n\n", "section_header")
        about_text = BREED_INFO.get(
            prediction['breed'],
            "Detailed information for this breed is not available yet."
        ) + GENERIC_BREED_NOTE
        self.results_text.insert(tk.END, about_text, "details")

        self.results_text.tag_config("icon", foreground=self.colors['success'], font=("Segoe UI", 12))
        self.results_text.tag_config("section_header", font=("Segoe UI", 11, "bold"),
                                     foreground=self.colors['text'])
        self.results_text.tag_config("breed_name", font=("Segoe UI", 18, "bold"),
                                     foreground=self.colors['primary'])
        self.results_text.tag_config("confidence", font=("Segoe UI", 10, "bold"),
                                     foreground=self.colors['accent'])
        self.results_text.tag_config("separator", foreground=self.colors['border'])
        self.results_text.tag_config("details", font=("Segoe UI", 9),
                                     foreground=self.colors['text_muted'])
        self.results_text.config(state=tk.DISABLED)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_item = {
            "breed": prediction['breed'],
            "confidence": prediction['confidence'],
            "timestamp": timestamp,
            "image_path": self.selected_image_path,
            "predictions": prediction['top_predictions']
        }
        self.history.insert(0, history_item)
        self.update_history_display()
        self.is_analyzing = False
        self.analyze_btn.config(state=tk.NORMAL, text="🔍  Identify Breed")
        self.upload_btn.config(state=tk.NORMAL)
        messagebox.showinfo(
            "Analysis Complete",
            f"✔ Identified as {prediction['breed']} with {prediction['confidence']}% confidence"
        )
        self.upload_again_btn.pack(fill=tk.X, pady=(15, 0))

    def reset_app(self):
        # Clear selection
        self.selected_image_path = None

        # Reset image label to placeholder state
        self.image_label.configure(image='', text='')
        self.upload_icon_label.pack(pady=(30, 10))
        self.upload_text_label.pack()
        self.image_container.configure(highlightbackground=self.colors['border'])

        # Reset card sizes and layout
        self.upload_card.configure(width=None, height=None)
        self.upload_card.pack_propagate(True)

        self.results_card.configure(width=None, height=None)
        self.results_card.pack_propagate(True)

        self.upload_again_btn.pack_forget()

        # Clear results text area
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

        # Enable Upload button, disable Analyze button
        self.upload_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.DISABLED, text="🔍  Identify Breed")

    def update_history_display(self):
        if self.history:
            self.history_empty_label.pack_forget()
            self.history_listbox.delete(0, tk.END)
            self.history_count_label.config(text=str(len(self.history)))
            for item in self.history:
                display_text = f"  {item['breed']} • {item['confidence']}%"
                self.history_listbox.insert(tk.END, display_text)
                self.history_listbox.insert(tk.END, f"  📅 {item['timestamp']}")
                self.history_listbox.insert(tk.END, "")
        else:
            self.history_empty_label.pack(expand=True)
            self.history_count_label.config(text="0")

    def on_history_select(self, event):
        selection = self.history_listbox.curselection()
        if selection:
            index = selection[0] // 3
            if index < len(self.history):
                item = self.history[index]

                self.upload_icon_label.pack_forget()
                self.upload_text_label.pack_forget()
                self.selected_image_path = item['image_path']

                # Try to load and display the image for this history item
                try:
                    img = Image.open(self.selected_image_path).convert("RGB")
                    img.thumbnail((400, 400))
                    self.tk_image = ImageTk.PhotoImage(img)
                    self.image_label.configure(
                        image=self.tk_image,
                        text="",
                        bg=self.colors['card_secondary'],
                        fg=self.colors['primary']
                    )
                    self.image_label.image = self.tk_image
                    self.image_container.configure(highlightbackground=self.colors['border_hover'])
                except Exception:
                    file_name = os.path.basename(self.selected_image_path)
                    self.image_label.configure(
                        image='',
                        text=file_name,
                        font=("Segoe UI", 13, "bold"),
                        bg=self.colors['card_secondary'],
                        fg=self.colors['primary']
                    )

                self.expand_results_shrink_upload()
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✔ ", "icon")
                self.results_text.insert(tk.END, "Primary Breed Identified\n\n", "section_header")
                self.results_text.insert(tk.END, f"{item['breed']}\n", "breed_name")
                self.results_text.insert(
                    tk.END,
                    f"● {item['confidence']}% confident\n\n",
                    "confidence"
                )
                self.results_text.insert(tk.END, "─" * 60 + "\n\n", "separator")
                self.results_text.insert(tk.END, f"ℹ  About {item['breed']}\n\n", "section_header")
                about_text = BREED_INFO.get(
                    item['breed'],
                    "Detailed information for this breed is not available yet."
                ) + GENERIC_BREED_NOTE
                self.results_text.insert(tk.END, about_text, "details")

                self.results_text.tag_config("icon", foreground=self.colors['success'], font=("Segoe UI", 12))
                self.results_text.tag_config("section_header", font=("Segoe UI", 11, "bold"),
                                             foreground=self.colors['text'])
                self.results_text.tag_config("breed_name", font=("Segoe UI", 18, "bold"),
                                             foreground=self.colors['primary'])
                self.results_text.tag_config("confidence", font=("Segoe UI", 10, "bold"),
                                             foreground=self.colors['accent'])
                self.results_text.tag_config("details", font=("Segoe UI", 9),
                                             foreground=self.colors['text_muted'])
                self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = BreedVisionApp(root)
    root.mainloop()
