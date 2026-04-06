import cv2
import tkinter as tk
from tkinter import messagebox, ttk
import time
import insightface
import numpy as np
import threading
from PIL import Image, ImageTk
import os
import faiss
import sys
from enum import Enum
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import subprocess
from dotenv import load_dotenv

load_dotenv()

# Função essencial para o executável (.exe) encontrar os arquivos
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))
    
class RecognitionState(Enum):
    AWAITING = 1
    PROCESSING = 2
    SHOWING_RESULT = 3

class FaceAccessControl:
    THRESHOLD = 0.55  # Rigor da IA (evita falsos positivos)
    PROCESSING_TIME_SECONDS = 1.5  # Tempo rápido de processamento

    def __init__(self, root):
        self.root = root
        
        # --- [ CONFIGURAÇÃO DO BANCO ] ---
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'database': os.getenv("DB_NAME")
        }
        self.db_table    = "move"
        self.db_col_nome = "mat"
        self.db_col_maq  = "maq"
        # ---------------------------------

        # --- [ MODO DE OPERAÇÃO DA MÁQUINA ] ---
        # 0 = ENTRADA
        # 1 = SAÍDA
        self.MODO_OPERACAO = 0  # <--- MUDE PARA 1 NO NOTEBOOK DE SAÍDA!
        # ---------------------------------------

        self.nome_acao = "ENTRADA" if self.MODO_OPERACAO == 0 else "SAÍDA"
        self.rota_arduino = "/entrada/abrir" if self.MODO_OPERACAO == 0 else "/saida/abrir"

        self.root.title(f"Controle de Acesso - {self.nome_acao}")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model = None
        self.faiss_index = None
        self.known_names = []
        self.cap = None
        
        # O executável vai criar essas pastas onde ele estiver rodando
        self.base_dir = get_base_path()
        self.gallery_path = os.path.join(self.base_dir, "galeria")
        self.cache_path = os.path.join(self.base_dir, "embeddings_cache")
        os.makedirs(self.gallery_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)
        
        self.current_state = RecognitionState.AWAITING
        self.processing_start_time = None
        self.best_shot_embedding = None
        self.best_shot_area = 0
        
        self.create_widgets()
        threading.Thread(target=self.init_model, daemon=True).start()

    def create_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        titulo = tk.Label(main_frame, text=f"CATRACA DE {self.nome_acao}", font=("Arial", 16, "bold"))
        titulo.pack(pady=5)
        
        self.video_label = tk.Label(main_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.update_button = ttk.Button(control_frame, text="Atualizar Galeria", command=self.update_gallery)
        self.update_button.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.quit_button = ttk.Button(control_frame, text="Encerrar", command=self.on_closing)
        self.quit_button.pack(side=tk.RIGHT, padx=5)
        
        self.status_label = tk.Label(main_frame, text="Inicializando sistema...", font=("Arial", 14))
        self.status_label.pack(pady=5)

    def init_model(self):
        try:
            self.model = insightface.app.FaceAnalysis(name="buffalo_l", root=self.base_dir)
            self.model.prepare(ctx_id=1)
            self.root.after(0, self.load_gallery)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao inicializar o modelo:\n{e}")
            self.on_closing()

    def registrar_no_banco(self, match_name):
        query = f"INSERT INTO {self.db_table} ({self.db_col_nome}, {self.db_col_maq}) VALUES (%s, %s)"
        args = (match_name, self.MODO_OPERACAO)
        connection = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            cursor.execute(query, args)
            connection.commit()
            print(f"[{self.nome_acao}] Registro salvo: {match_name}")
        except Error as e:
            print(f"Erro no MySQL: {e}")
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    def abrir_catraca(self):
        ip_da_catraca = "192.168.4.1"
        try:
            comando = ["curl", "-s", f"http://{ip_da_catraca}{self.rota_arduino}"]
            subprocess.Popen(comando)   
            print(f"Sinal de {self.nome_acao} enviado!")
        except Exception as e:
            print(f"Erro na catraca: {e}")

    def load_gallery(self):
        self.status_label.config(text="Carregando galeria...", fg="black")
        threading.Thread(target=self._process_gallery, args=(False,), daemon=True).start()

    def update_gallery(self):
        self.update_button.config(state=tk.DISABLED)
        self.status_label.config(text="Atualizando galeria...", fg="blue")
        threading.Thread(target=self._process_gallery, args=(True,), daemon=True).start()

    def _process_gallery(self, only_new):
        files = [f for f in os.listdir(self.gallery_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if only_new:
            files = [f for f in files if not os.path.exists(os.path.join(self.cache_path, f"{os.path.splitext(f)[0]}.npy"))]
        if not files and only_new:
            self.root.after(0, self._gallery_updated)
            return
            
        embeddings, names = [], []
        if not only_new:
            for f in [f for f in os.listdir(self.gallery_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]:
                name, cache = os.path.splitext(f)[0], os.path.join(self.cache_path, f"{os.path.splitext(f)[0]}.npy")
                if os.path.exists(cache):
                    embeddings.append(np.load(cache))
                    names.append(name)
                    
        self.progress['maximum'] = len(files)
        for i, f in enumerate(files):
            name, cache = os.path.splitext(f)[0], os.path.join(self.cache_path, f"{name}.npy")
            img = cv2.imread(os.path.join(self.gallery_path, f))
            if img is None: continue
            
            # Reduz o tamanho da foto da galeria para processar rápido
            img_resized = cv2.resize(img, (640, 480))
            faces = self.model.get(img_resized)
            if not faces: continue
            
            emb = faces[0].embedding
            np.save(cache, emb)
            embeddings.append(emb)
            names.append(name)
            self.progress['value'] = i + 1
            self.root.update_idletasks()
            
        if embeddings:
            emb_np = np.array(embeddings).astype('float32')
            faiss.normalize_L2(emb_np)
            self.faiss_index = faiss.IndexFlatIP(emb_np.shape[1])
            self.faiss_index.add(emb_np)
            self.known_names = names
            
        self.root.after(0, self._gallery_updated)

    def _gallery_updated(self):
        self.progress['value'] = 0
        self.status_label.config(text=f"Pronto! ({self.faiss_index.ntotal if self.faiss_index else 0} rostos)", fg="green")
        self.update_button.config(state=tk.NORMAL)
        if not self.cap: self.start_camera()

    def start_camera(self):
        # 0 é a câmera com fio (USB). Se for usar a segunda porta USB, pode ser 1.
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a câmera USB.")
            self.on_closing()
            return
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # NITRO: Diminui a imagem da câmera para a IA processar mais rápido!
            small_frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = rgb.copy()
            
            if self.faiss_index and self.faiss_index.ntotal > 0 and self.current_state != RecognitionState.SHOWING_RESULT:
                faces = self.model.get(small_frame) # A IA analisa a imagem menor
                
                if faces:
                    main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    
                    # Desenha o quadrado escalado para o tamanho original na tela
                    scale_y, scale_x = frame.shape[0] / 480, frame.shape[1] / 640
                    bbox = main_face.bbox.astype(int)
                    x1, y1 = int(bbox[0] * scale_x), int(bbox[1] * scale_y)
                    x2, y2 = int(bbox[2] * scale_x), int(bbox[3] * scale_y)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if self.current_state == RecognitionState.AWAITING:
                        self.current_state = RecognitionState.PROCESSING
                        self.processing_start_time = time.time()
                        self.best_shot_area = 0
                        self.best_shot_embedding = None
                        self.status_label.config(text="Analisando face...", fg="blue")
                        
                    if self.current_state == RecognitionState.PROCESSING:
                        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if face_area > self.best_shot_area:
                            self.best_shot_area = face_area
                            self.best_shot_embedding = main_face.embedding
                            
                        if time.time() - self.processing_start_time >= self.PROCESSING_TIME_SECONDS:
                            self.process_best_shot()
                elif self.current_state == RecognitionState.PROCESSING: 
                    self.reset_state()
            
            pil_img = Image.fromarray(processed_frame)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(30, self.update_frame)

    def process_best_shot(self):
        self.current_state = RecognitionState.SHOWING_RESULT
        if self.best_shot_embedding is not None:
            emb = self.best_shot_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(emb)
            D, I = self.faiss_index.search(emb, 1)
            
            if D[0][0] >= self.THRESHOLD:
                match_name = self.known_names[I[0][0]]
                self.status_label.config(text=f"Acesso Liberado: {match_name}", fg="green")
                self.registrar_no_banco(match_name)
                self.abrir_catraca()
            else: 
                self.status_label.config(text="Acesso Negado (Rosto Desconhecido)", fg="red")
                
            self.root.after(2000, self.reset_state)
        else: 
            self.reset_state()
    
    def reset_state(self):
        self.current_state = RecognitionState.AWAITING
        self.processing_start_time = None
        self.best_shot_area = 0
        self.best_shot_embedding = None
        self.status_label.config(text="Aguardando face...", fg="black")

    def on_closing(self):
        if self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAccessControl(root)
    root.mainloop()
