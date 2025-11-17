import zipfile
import os
import time
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from gerar_relatorio import gerar_relatorio_pdf 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
ZIP_PATH = "dataset_test.zip"
EXTRACT_PATH = "dataset_extracted"
SERVER_URL = "http://127.0.0.1:5000/recognize"
CLASSES = ["jorge", "felipe", "brito", "unknown"]

# -----------------------------
# EXTRAIR ARQUIVOS ZIP
# -----------------------------
def extract_zip():
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)

        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

    print(f"[OK] Dataset extraído em: {EXTRACT_PATH}")

# -----------------------------
# IDENTIFICAR CLASSE ESPERADA
# Pelo nome da pasta ou arquivo
# -----------------------------
def get_expected_class(img_path):
    img_lower = img_path.lower()

    for c in CLASSES:
        if c in img_lower:
            return c

    # Se não tiver nenhuma classe no nome → unknown esperado
    return "unknown"

# -----------------------------
# ENVIAR IMAGEM PARA O SERVIDOR
# -----------------------------
def send_image(img_path):
  with open(img_path, "rb") as f:
      img_bytes = f.read()

      start = time.time()
      response = requests.post(
          SERVER_URL,
          data=img_bytes,                     # <- ENVIO CORRETO
          headers={"Content-Type": "image/jpeg"}
      )
      latency = (time.time() - start) * 1000  # ms

  return response.json(), latency

# -----------------------------
# VARREDURA DE TODAS AS IMAGENS
# -----------------------------
def run_tests():
    registros = []  # <-- armazenar para PDF

    y_true = []
    y_pred = []
    latencies = []

    for root, _, files in os.walk(EXTRACT_PATH):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, file)
            expected = get_expected_class(img_path)

            result, latency = send_image(img_path)

            recognized = result.get("recognized", False)
            identity = result.get("identity", "unknown")
            confidence = result.get("confidence", None)

            cpu_before = result.get("cpu_before", None)
            cpu_after = result.get("cpu_after", None)
            ram_before = result.get("ram_before", None)  # bytes
            ram_after = result.get("ram_after", None)

            y_true.append(expected)
            y_pred.append(identity)
            latencies.append(latency)

            registros.append({
                "imagem": img_path,
                "classe_real": expected,
                "predicao": identity,
                "confidence": confidence,
                "latencia_ms": latency,
                "cpu_before": cpu_before,
                "cpu_after": cpu_after,
                "ram_before_mb": ram_before / (1024 * 1024) if ram_before else None,
                "ram_after_mb":  ram_after  / (1024 * 1024) if ram_after  else None,
            })

            print(f"[IMG] {file} | real={expected} | pred={identity} | {round(latency,2)}ms")

    df = pd.DataFrame(registros)
    return y_true, y_pred, latencies, df

# -----------------------------
# MÉTRICAS BIOMÉTRICAS
# -----------------------------
def biometric_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classes reais (sem unknown/vazio)
    is_biometric = y_true != "vazio"

    genuine = (y_true == y_pred) & is_biometric
    impostor = (y_true != y_pred) & is_biometric

    TP = genuine.sum()
    FN = (is_biometric.sum() - TP)
    FP = impostor.sum()
    TN = 0  # não usado em biometria clássica

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FRR = FN / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (FP + TP) if (FP + TP) > 0 else 0

    # EER ≈ ponto onde FAR ~= FRR
    EER = abs(FAR - FRR)

    return {
        "TPR/GAR": TPR,
        "FAR": FAR,
        "FRR": FRR,
        "EER (aprox)": EER,
        "TP": TP,
        "FN": FN,
        "FP": FP
    }

def gerar_roc_auc(df, output_path="../tmp/roc_curve.png"):
    """
    Gera curva ROC e salva em PNG.
    df deve ter 'classe_real', 'predicao' e 'confidence'
    """
    # classes biométricas reais (remove 'vazio' e 'unknown')
    df_valid = df[df["classe_real"] != "vazio"]
    df_valid = df_valid[df_valid["confidence"].notna()]

    if len(df_valid) < 2:
        print("[WARN] Dados insuficientes para ROC/AUC.")
        return None

    # classe positiva: match correto
    y_true = (df_valid["classe_real"] == df_valid["predicao"]).astype(int)
    y_score = df_valid["confidence"].astype(float)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")  # linha aleatória
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[OK] Curva ROC gerada: {output_path}")
    return output_path, roc_auc

def gerar_grafico_cpu_ram(df, output_path="../tmp/cpu_ram_timeline.png"):
    plt.figure(figsize=(8, 5))

    plt.plot(df.index, df["cpu_after"], label="CPU (%)")
    plt.plot(df.index, df["ram_after_mb"], label="RAM (MB)")

    plt.xlabel("Inferência")
    plt.ylabel("Uso de recursos")
    plt.title("Uso de CPU e RAM ao longo do tempo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[OK] Gráfico CPU/RAM gerado: {output_path}")

    return output_path

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    extract_zip()

    print("\n[TESTES] Enviando imagens para o servidor...")
    y_true, y_pred, latencies, resultados_df = run_tests()

    print("\n---------------- MÉTRICAS GERAIS ----------------")
    print("Accuracy:", np.mean(np.array(y_true) == np.array(y_pred)))
    latencies = latencies[1:]
    print("Latência média:", np.mean(latencies), "ms")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=CLASSES))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=CLASSES))

    print("\n---------------- MÉTRICAS BIOMÉTRICAS ----------------")
    bio = biometric_metrics(y_true, y_pred)
    for k, v in bio.items():
        print(f"{k}: {v}")
    
    # gerar graficos
    roc_info = gerar_roc_auc(resultados_df)
    cpu_ram_path = gerar_grafico_cpu_ram(resultados_df)
    if roc_info is not None:
        roc_path, roc_auc = roc_info
    else:
        roc_path, roc_auc = None, None

    gerar_relatorio_pdf(
        resultados_df,
        y_true,
        y_pred,
        latencies,
        bio,
        roc_path=roc_path,
        roc_auc=roc_auc,
        cpu_ram_path=cpu_ram_path,
        output_path="relatorio_resultados.pdf",
        classes=CLASSES
    )
