from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def gerar_relatorio_pdf(
    df,
    y_true,
    y_pred,
    latencies,
    biometric,
    roc_path=None,
    roc_auc=None,
    cpu_ram_path=None,
    output_path="relatorio_resultados.pdf",
    classes=None
):
    """
    df: dataframe com (imagem, classe_real, predicao, confidence, latencia_ms)
    y_true: lista de classes reais
    y_pred: lista de predições
    latencies: lista de latências
    biometric: dicionário das métricas biométricas
    roc_path : caminho para a imagem PNG contendo a curva ROC
    roc_auc : valor da AUC (Área sob a curva ROC)
    classes: lista de classes para matriz de confusão
    """

    styles = getSampleStyleSheet()
    story = []

    # --------------------------
    # TÍTULO
    # --------------------------
    titulo = Paragraph("<b>Relatório Completo de Testes de Reconhecimento Facial</b>", styles["Title"])
    story.append(titulo)
    story.append(Spacer(1, 20))

    # --------------------------
    # MÉTRICAS GERAIS
    # --------------------------
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    latency_avg = np.mean(latencies)

    txt_metricas = f"""
    <b>Métricas Gerais</b><br/>
    Accuracy: {accuracy:.4f}<br/>
    Latência média: {latency_avg:.2f} ms<br/><br/>
    """

    story.append(Paragraph(txt_metricas, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Métricas de Recursos</b>", styles["Heading2"]))

    cpu_media = df["cpu_after"].mean()
    ram_media = df["ram_after_mb"].mean()
    ram_pico = df["ram_after_mb"].max()

    texto_recursos = f"""
    CPU média: {cpu_media:.2f}%<br/>
    RAM média: {ram_media:.2f} MB<br/>
    Pico de RAM: {ram_pico:.2f} MB<br/>
    """

    story.append(Paragraph(texto_recursos, styles["Normal"]))
    story.append(Spacer(1, 20))

    # --------------------------
    # MÉTRICAS BIOMÉTRICAS
    # --------------------------
    txt_bio = "<b>Métricas Biométricas</b><br/>"
    for k, v in biometric.items():
        txt_bio += f"{k}: {v:.4f}<br/>"

    story.append(Paragraph(txt_bio, styles["Normal"]))
    story.append(Spacer(1, 20))

    # --------------------------
    # MATRIZ DE CONFUSÃO
    # --------------------------
    story.append(Paragraph("<b>Matriz de Confusão</b>", styles["Heading2"]))

    if classes is None:
        classes = sorted(list(set(y_true + y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    tabela_cm = [[""] + classes]  # header
    for i, row in enumerate(cm):
        tabela_cm.append([classes[i]] + list(row))

    tabela = Table(tabela_cm)
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
    ]))
    story.append(tabela)
    story.append(Spacer(1, 20))

    # --------------------------
    # CLASSIFICATION REPORT
    # --------------------------
    story.append(Paragraph("<b>Classification Report</b>", styles["Heading2"]))

    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    report_df = pd.DataFrame(report).T.reset_index()

    tabela_rep = [report_df.columns.tolist()] + report_df.values.tolist()

    tabela2 = Table(tabela_rep)
    tabela2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
    ]))
    story.append(tabela2)

    story.append(PageBreak())

    # --------------------------
    # GRAFICOS
    # --------------------------    
    if roc_path is not None:
        story.append(Paragraph("<b>Curva ROC</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))
        story.append(Image(roc_path, width=350, height=350))
        story.append(Spacer(1, 20))

        story.append(Paragraph(f"AUC = {roc_auc:.4f}", styles["Normal"]))
        story.append(Spacer(1, 30))

    if cpu_ram_path is not None:
        story.append(Paragraph("<b>Uso de CPU e RAM ao longo do tempo</b>", styles["Heading2"]))
        story.append(Image(cpu_ram_path, width=350, height=300))
        story.append(Spacer(1, 20))

    story.append(PageBreak())

    # --------------------------
    # SEÇÃO DE IMAGENS
    # --------------------------
    story.append(Paragraph("<b>Resultados por Imagem</b>", styles["Heading1"]))
    story.append(Spacer(1, 20))

    for idx, row in df.iterrows():
        img_path = row["imagem"]

        # Adicionar imagem
        story.append(Image(img_path, width=224, height=224))
        story.append(Spacer(1, 10))

        # Tabela com dados da imagem
        tabela_dados = [
            ["Campo", "Valor"],
            ["Classe real", row["classe_real"]],
            ["Predição", row["predicao"]],
            ["Confidence", str(row["confidence"])],
            ["Latência (ms)", round(row["latencia_ms"], 2)],
        ]

        tabela3 = Table(tabela_dados, colWidths=[120, 300])
        tabela3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
        ]))

        story.append(tabela3)
        story.append(Spacer(1, 25))

    # --------------------------
    # GERAR PDF
    # --------------------------
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    doc.build(story)

    print(f"[OK] Relatório PDF gerado: {output_path}")
