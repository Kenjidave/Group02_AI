import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import csv

# --- Configuration & Theme ---
DB_NAME = 'diapredict.db'
MODEL_FILE = 'diapredict_best_model.pkl'
SCALER_FILE = 'scaler.pkl'

BG_COLOR = "#F4F3EE"
CARD_BG = "#FFFFFF"
PRIMARY_BLUE = "#4A90E2"
TEXT_MAIN = "#333333"
TEXT_LIGHT = "#777777"

class DiaPredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DiaPredict - Diabetes Risk Assessment")
        self.root.geometry("450x750")
        self.root.configure(bg=BG_COLOR)
        
        self.current_user_id = None
        self.current_user_name = "User"
        self.current_frame = None

        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found.")
            self.root.destroy()
            return

        self.setup_styles()
        self.show_login_screen()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TEntry", fieldbackground=CARD_BG, padding=8, borderwidth=0)
        style.configure("Treeview", rowheight=30, background=CARD_BG, fieldbackground=CARD_BG)
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), background=BG_COLOR)

    def clear_frame(self):
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    
    # --- SCREEN 1: Login ---
    def show_login_screen(self):
        self.clear_frame()
        tk.Label(self.current_frame, text="DiaPredict", font=("Helvetica", 24, "bold"), bg=BG_COLOR, fg=TEXT_MAIN).pack(pady=(40, 5))
        tk.Label(self.current_frame, text="Diabetes risk assessment", font=("Helvetica", 12), bg=BG_COLOR, fg=TEXT_LIGHT).pack(pady=(0, 40))

        tk.Label(self.current_frame, text="Email address", bg=BG_COLOR, fg=TEXT_MAIN, font=("Helvetica", 10)).pack(anchor="w")
        email_entry = ttk.Entry(self.current_frame, font=("Helvetica", 12))
        email_entry.pack(fill="x", pady=(0, 15))

        tk.Label(self.current_frame, text="Password", bg=BG_COLOR, fg=TEXT_MAIN, font=("Helvetica", 10)).pack(anchor="w")
        password_entry = ttk.Entry(self.current_frame, show="*", font=("Helvetica", 12))
        password_entry.pack(fill="x", pady=(0, 30))

        def attempt_login():
            email, password = email_entry.get().strip(), password_entry.get().strip()
            with sqlite3.connect(DB_NAME) as conn:
                user = conn.cursor().execute("SELECT user_id, name FROM users WHERE email=? AND password=?", (email, password)).fetchone()
            if user:
                self.current_user_id, self.current_user_name = user[0], user[1]
                self.show_assessment_screen()
            else: messagebox.showerror("Error", "Invalid credentials.")

        def create_account():
            email, password = email_entry.get().strip(), password_entry.get().strip()
            if not email or not password: return
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    conn.cursor().execute("INSERT INTO users VALUES (NULL, ?, ?, ?)", (email, password, email.split('@')[0].capitalize()))
                    conn.commit()
                messagebox.showinfo("Success", "Account created!")
            except: messagebox.showerror("Error", "Account exists.")

        tk.Button(self.current_frame, text="Sign in", command=attempt_login, bg=CARD_BG, fg=TEXT_MAIN, font=("Helvetica", 12, "bold"), relief="solid", bd=1, pady=8).pack(fill="x", pady=10)
        tk.Button(self.current_frame, text="Create account", command=create_account, bg=CARD_BG, fg=TEXT_MAIN, font=("Helvetica", 12), relief="solid", bd=1, pady=8).pack(fill="x")

    def logout(self):
        """Clears the current session and returns to the login screen."""
        self.current_user_id = None
        self.current_user_name = "User"
        self.show_login_screen()

    
    # --- SCREEN 2: Health Form (Centered Layout) ---
    def show_assessment_screen(self):
        self.clear_frame()
        
        # Header
        header_frame = tk.Frame(self.current_frame, bg=BG_COLOR)
        header_frame.pack(fill="x", pady=(0, 20))
        tk.Label(header_frame, text="New assessment", font=("Helvetica", 16, "bold"), bg=BG_COLOR, fg=TEXT_MAIN).pack(side=tk.LEFT)
        
        # User Badge
        badge = tk.Label(header_frame, text=self.current_user_name, font=("Helvetica", 10), bg="#E6F0FA", fg=PRIMARY_BLUE, padx=10, pady=5)
        badge.pack(side=tk.RIGHT)

        tk.Label(self.current_frame, text="Enter health metrics", font=("Helvetica", 12, "bold"), bg=BG_COLOR, fg=TEXT_MAIN).pack(anchor="center", pady=(10, 5))

        self.inputs = {}
        features = [
            ("Glucose (mg/dL)", "Glucose"), ("BMI", "BMI"),
            ("Blood pressure", "BloodPressure"), ("Age", "Age"),
            ("Insulin (μU/mL)", "Insulin"), ("Skin thickness", "SkinThickness"),
            ("Pregnancies", "Pregnancies"), ("DPF score", "DiabetesPedigreeFunction")
        ]

        # Container for the grid, centered using pack
        grid_container = tk.Frame(self.current_frame, bg=BG_COLOR)
        grid_container.pack(anchor="center", pady=10)

        # Configure grid columns to be centered
        grid_container.grid_columnconfigure(0, weight=1)
        grid_container.grid_columnconfigure(1, weight=1)

        for i, (label_text, feature_name) in enumerate(features):
            row, col = i // 2, i % 2
            
            # Individual input frame
            frame = tk.Frame(grid_container, bg=BG_COLOR)
            frame.grid(row=row, column=col, padx=15, pady=8)
            
            tk.Label(frame, text=label_text, bg=BG_COLOR, fg=TEXT_MAIN, font=("Helvetica", 9)).pack(anchor="w")
            entry_var = tk.StringVar()
            ttk.Entry(frame, textvariable=entry_var, font=("Helvetica", 11), width=18).pack(fill="x")
            self.inputs[feature_name] = entry_var

        tk.Button(self.current_frame, text="Run assessment", command=self.process_assessment, bg=CARD_BG, fg=TEXT_MAIN, font=("Helvetica", 12, "bold"), relief="solid", bd=1, pady=10).pack(fill="x", pady=(20, 10))
        tk.Button(self.current_frame, text="View History", command=self.show_history_screen, bg=BG_COLOR, fg=PRIMARY_BLUE, font=("Helvetica", 10, "underline"), relief="flat").pack()
        tk.Button(self.current_frame, text="Sign out", command=self.logout, bg=BG_COLOR, fg="#E74C3C", font=("Helvetica", 10, "underline"), relief="flat").pack(pady=(5, 0))

    
    # --- SCREEN 3: Result Display ---
    def process_assessment(self):
        try:
            ordered_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            user_data = [float(self.inputs[f].get().strip()) for f in ordered_features]
            
            user_data_df = pd.DataFrame([user_data], columns=ordered_features)
            user_data_scaled = self.scaler.transform(user_data_df)
            probability = self.model.predict_proba(user_data_scaled)[0][1]
            
            if probability < 0.33:
                risk_level, color, bg_color = "Low risk", "#155724", "#D4EDDA"
            elif probability < 0.66:
                risk_level, color, bg_color = "Medium risk", "#856404", "#FFF3CD"
            else:
                risk_level, color, bg_color = "High risk", "#721C24", "#F8D7DA"

            self.show_result_screen(probability, risk_level, color, bg_color, user_data)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")

    def show_result_screen(self, probability, risk_level, color, bg_color, user_data):
        self.clear_frame()
        
        # --- Header ---
        header_frame = tk.Frame(self.current_frame, bg=BG_COLOR)
        header_frame.pack(fill="x", pady=(0, 20))
        tk.Label(header_frame, text="Assessment result", font=("Helvetica", 16, "bold"), bg=BG_COLOR, fg=TEXT_MAIN).pack(side=tk.LEFT)
        tk.Label(header_frame, text=self.current_user_name, font=("Helvetica", 10), bg="#E6F0FA", fg=PRIMARY_BLUE, padx=10, pady=5).pack(side=tk.RIGHT)

        # --- Big Result Card ---
        result_card = tk.Frame(self.current_frame, bg=CARD_BG, relief="flat", bd=0, padx=20, pady=30)
        result_card.pack(fill="x", pady=10)
        
        tk.Label(result_card, text=f"{probability*100:.0f}%", font=("Helvetica", 36, "bold"), bg=CARD_BG, fg=TEXT_MAIN).pack()
        tk.Label(result_card, text="Diabetes risk probability", font=("Helvetica", 12), bg=CARD_BG, fg=TEXT_LIGHT).pack(pady=(0, 10))
        tk.Label(result_card, text=risk_level, font=("Helvetica", 10, "bold"), bg=bg_color, fg=color, padx=15, pady=5).pack()

        # --- Input Summary Box ---
        summary_card = tk.Frame(self.current_frame, bg=CARD_BG, relief="flat", padx=15, pady=15)
        summary_card.pack(fill="x", pady=10)
        
        tk.Label(summary_card, text="Input summary", font=("Helvetica", 12, "bold"), bg=CARD_BG, fg=TEXT_MAIN).pack(anchor="w", pady=(0, 10))
        
        display_data = [
            ("Glucose", f"{user_data[1]:.0f} mg/dL"), ("BMI", f"{user_data[5]:.1f}"),
            ("Age", f"{user_data[7]:.0f}"), ("Blood pressure", f"{user_data[2]:.0f} mmHg")
        ]
        
        for label, val in display_data:
            row_frame = tk.Frame(summary_card, bg=CARD_BG)
            row_frame.pack(fill="x", pady=2)
            tk.Label(row_frame, text=label, bg=CARD_BG, fg=TEXT_LIGHT, font=("Helvetica", 10)).pack(side=tk.LEFT)
            tk.Label(row_frame, text=val, bg=CARD_BG, fg=TEXT_MAIN, font=("Helvetica", 10, "bold")).pack(side=tk.RIGHT)

        # --- Warning & Save Button ---
        tk.Label(self.current_frame, text="⚠ Consult a healthcare professional for a formal diagnosis.", 
                 font=("Helvetica", 9), bg="#F8D7DA", fg="#721C24", wraplength=400, pady=10).pack(fill="x", pady=10)
        
        def save_and_return():
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    conn.cursor().execute('''INSERT INTO assessments 
                        (user_id, assessment_date, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, risk_level, probability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                        (self.current_user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *user_data, risk_level, probability))
                    conn.commit()
            except Exception as e: print(e)
            self.show_assessment_screen()

        tk.Button(self.current_frame, text="Save to history", command=save_and_return, bg=CARD_BG, fg=TEXT_MAIN, 
                  font=("Helvetica", 12, "bold"), relief="solid", bd=1, pady=10).pack(fill="x")

    
        
    # --- SCREEN 4: History ---
    def show_history_screen(self):
        self.clear_frame()
        
        # Header Area
        header_frame = tk.Frame(self.current_frame, bg=BG_COLOR)
        header_frame.pack(fill="x", pady=(0, 10))
        tk.Label(header_frame, text="Assessment history", font=("Helvetica", 16, "bold"), bg=BG_COLOR, fg=TEXT_MAIN).pack(side=tk.LEFT)
        tk.Label(header_frame, text=self.current_user_name, font=("Helvetica", 10), bg="#E6F0FA", fg=PRIMARY_BLUE, padx=10, pady=5).pack(side=tk.RIGHT)

        # Table Area
        tree_frame = tk.Frame(self.current_frame, bg=BG_COLOR)
        tree_frame.pack(fill="both", expand=True, pady=10)
        
        tree = ttk.Treeview(tree_frame, columns=("Date", "Glucose", "BMI", "Risk"), show="headings", height=10)
        tree.heading("Date", text="Date"); tree.column("Date", width=100, anchor="center")
        tree.heading("Glucose", text="Glucose"); tree.column("Glucose", width=70, anchor="center")
        tree.heading("BMI", text="BMI"); tree.column("BMI", width=50, anchor="center")
        tree.heading("Risk", text="Risk"); tree.column("Risk", width=80, anchor="center")
        tree.pack(fill="both", expand=True)

        # Add data with color tags
        tree.tag_configure("Low risk", background="#D4EDDA", foreground="#155724")
        tree.tag_configure("Medium risk", background="#FFF3CD", foreground="#856404")
        tree.tag_configure("High risk", background="#F8D7DA", foreground="#721C24")
        
        try:
            with sqlite3.connect(DB_NAME) as conn:
                for r in conn.cursor().execute('SELECT * FROM assessments WHERE user_id=? ORDER BY assessment_date DESC', (self.current_user_id,)):
                    # DB Mapping: r[2]=Date, r[4]=Glucose, r[8]=BMI, r[11]=Risk
                    tree.insert("", "end", values=(r[2][:10], int(r[4]), round(float(r[8]), 1), r[11]), tags=(r[11],))
        except Exception as e: print(f"Error loading history: {e}")

        # Buttons (Exact Layout)
        def export():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", 
                initialfile=f"diapredict_history_{datetime.now().strftime('%Y%m%d')}.csv",
                filetypes=[("CSV Files", "*.csv")]
            )
            if path:
                with sqlite3.connect(DB_NAME) as conn, open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["Date", "Glucose", "BMI", "Blood Pressure", "Age", "Insulin", "Skin Thickness", "Pregnancies", "DPF Score", "Risk Level", "Probability"])
                    for r in conn.cursor().execute('SELECT * FROM assessments WHERE user_id=? ORDER BY assessment_date DESC', (self.current_user_id,)):
                        # Mapping: Date, Gluc, BMI, BP, Age, Ins, Skin, Preg, DPF, Risk, Prob
                        w.writerow([r[2], r[4], r[8], r[5], r[10], r[7], r[6], r[3], r[9], r[11], r[12]])
                messagebox.showinfo("Export", "Success!")

        tk.Button(self.current_frame, text="New assessment", command=self.show_assessment_screen, bg=CARD_BG, relief="solid", bd=1, pady=10).pack(fill="x", pady=(5, 5))
        tk.Button(self.current_frame, text="Export records", command=export, bg=BG_COLOR, fg=PRIMARY_BLUE, relief="solid", bd=1, pady=10).pack(fill="x", pady=(0, 5))
        tk.Button(self.current_frame, text="Sign out", command=self.logout, bg=BG_COLOR, fg="#E74C3C", font=("Helvetica", 10, "underline"), relief="flat").pack()

if __name__ == "__main__":
    root = tk.Tk()
    DiaPredictApp(root)
    root.mainloop()