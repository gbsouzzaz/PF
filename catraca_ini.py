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
from dotenv import load_dotenv


load_dotenv()


class SistemaCatraca:
    def __init__(self):

        # --- CONFIGURAÇÃO DO SISTEMA ---
        self.CAMERA_INDEX = 0
        self.THRESHOLD_RECONHECIMENTO = 0.55
        self.MODO_OPERACAO = 1  # 0 = Entrada | 1 = Saída
        self.INTERVALO_MINIMO = 5

        # --- CONFIGURAÇÃO DO BANCO ---
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'database': os.getenv("DB_NAME")
        }

        # --- DIRETÓRIOS ---
        self.pasta_galeria = "galeria"
        self.pasta_embeddings = "embeddings_cache"

        if not os.path.exists(self.pasta_embeddings):
            os.makedirs(self.pasta_embeddings)

        # --- VARIÁVEIS ---
        self.index_faiss = None
        self.ids = []
        self.ultimo_registro = {}

        # --- MODELO DE RECONHECIMENTO ---
        self.modelo = insightface.app.FaceAnalysis()
        self.modelo.prepare(ctx_id=0)

        # --- CÂMERA ---
        self.cap = cv2.VideoCapture(self.CAMERA_INDEX)

        # --- CARREGAR EMBEDDINGS ---
        self.carregar_galeria()

    def conectar_banco(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except Error as e:
            print("Erro ao conectar ao banco:", e)
            return None

    def carregar_galeria(self):
        embeddings = []

        for arquivo in os.listdir(self.pasta_galeria):
            caminho = os.path.join(self.pasta_galeria, arquivo)

            if arquivo.lower().endswith((".jpg", ".png", ".jpeg")):
                imagem = cv2.imread(caminho)

                faces = self.modelo.get(imagem)

                if len(faces) > 0:
                    embedding = faces[0].embedding
                    embeddings.append(embedding)
                    self.ids.append(os.path.splitext(arquivo)[0])

        if len(embeddings) > 0:
            embeddings = np.array(embeddings).astype("float32")

            dim = embeddings.shape[1]

            self.index_faiss = faiss.IndexFlatL2(dim)
            self.index_faiss.add(embeddings)

    def registrar_acesso(self, mat):

        agora = time.time()

        if mat in self.ultimo_registro:
            if agora - self.ultimo_registro[mat] < self.INTERVALO_MINIMO:
                return

        conn = self.conectar_banco()

        if conn is None:
            return

        cursor = conn.cursor()

        try:
            sql = "INSERT INTO move (mat, maq) VALUES (%s, %s)"
            valores = (mat, self.MODO_OPERACAO)

            cursor.execute(sql, valores)
            conn.commit()

            self.ultimo_registro[mat] = agora

            print(f"Acesso registrado: {mat}")

        except Error as e:
            print("Erro ao registrar acesso:", e)

        finally:
            cursor.close()
            conn.close()

    def reconhecer(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                continue

            faces = self.modelo.get(frame)

            for face in faces:

                bbox = face.bbox.astype(int)
                embedding = face.embedding.astype("float32")

                if self.index_faiss is None:
                    continue

                D, I = self.index_faiss.search(np.array([embedding]), 1)

                distancia = D[0][0]

                if distancia < self.THRESHOLD_RECONHECIMENTO:

                    mat = self.ids[I[0][0]]

                    self.registrar_acesso(mat)

                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.putText(
                        frame,
                        mat,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 0, 255),
                        2
                    )

            cv2.imshow("Sistema de Catraca", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    sistema = SistemaCatraca()

    sistema.reconhecer()