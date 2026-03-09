import insightface

print("Iniciando modelo...")

app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

print("MODELO CARREGADO COM SUCESSO!")