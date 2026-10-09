"""
main.py
=======
Script de integração do pipeline completo de diagnóstico preditivo:

    Dados Reais (PostgreSQL: Solis + Solarman) ou Sintéticos (Mock)
        ↓
    Detecção de Anomalias — Isolation Forest por inversor
        ↓
    Sumário Estruturado (AnomaliaSumario)
        ↓
    Pipeline RAG — Recuperação de contexto + Geração de Pré-Laudo
        ↓
    Laudos Técnicos Salvos em outputs/laudos/

PONTOS DE SUBSTITUIÇÃO marcados com # ← INTEGRAÇÃO REAL
"""

import sys
from pathlib import Path

# Garante que src/ é encontrado independente de onde main.py é executado
sys.path.insert(0, str(Path(__file__).parent))

from src.data.mock_generator import generate_synthetic_data, AnomalyEvent
from src.data.real_loader import load_canonical_dataframe
from src.models.anomaly_detector import AnomalyDetector
from src.rag.rag_pipeline import RAGPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Configurações Globais
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Dados
    "fonte_dados":       "real",  # "real" (PostgreSQL via túnel SSH) ou "mock"
    #"fonte_dados": "mock",

    # Dados reais — janela recomendada: dez/2025 a mai/2026 (fim exclusivo)
    "data_inicio":       "2025-12-01",
    "data_fim":          "2026-06-01",
    "incluir_solis":     True,
    "incluir_solarman":  True,

    # Dados mock
    "n_inversores":      5,
    "n_dias":            7,
    "seed":              42,

    # Isolation Forest
    "contamination":     0.05,    # ~5% de anomalias esperadas — ajustar nos dados reais
    "n_estimators":      100,

    # RAG
    "pasta_manuais":     "data/manuais",
    "pasta_laudos":      "outputs/laudos",
    "usar_mock_llm":     True,    # ← INTEGRAÇÃO REAL: defina False e configure _gerar_com_llm()
    "max_laudos_demo":   3,       # Gera laudos apenas dos N primeiros eventos (demo)
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  SISTEMA DE DIAGNÓSTICO PREDITIVO — INVERSORES FV")
    print("  PFC2 | EMC / UFG | Fábio Siqueira & Matheus Fernandes")
    print("=" * 65)

    # ── ETAPA 1: Dados ───────────────────────────────────────────────────────
    print("\n[ETAPA 1/3] Carregando dados...")

    if CONFIG["fonte_dados"] == "real":
        # Consultas em sql/ (Solis e Solarman); requer túnel SSH e .env
        df = load_canonical_dataframe(
            data_inicio=CONFIG["data_inicio"],
            data_fim=CONFIG["data_fim"],
            incluir_solis=CONFIG["incluir_solis"],
            incluir_solarman=CONFIG["incluir_solarman"],
        )
    else:
        df = generate_synthetic_data(
            n_inversores=CONFIG["n_inversores"],
            n_dias=CONFIG["n_dias"],
            seed=CONFIG["seed"],
        )

    print(f"\n  Resumo do dataset:")
    print(f"  • Registros: {len(df):,}")
    print(f"  • Inversores: {df['inversor_id'].unique().tolist()}")
    print(f"  • Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
    if "anomalia_real" in df.columns:
        print(f"  • Anomalias reais injetadas: "
              f"{(df['anomalia_real'] != 'normal').sum()} janelas "
              f"(coluna 'anomalia_real' existe apenas no mock)")

    # ── ETAPA 2: Detecção de Anomalias ───────────────────────────────────────
    print("\n[ETAPA 2/3] Detecção de anomalias (Isolation Forest)...")

    detector = AnomalyDetector(
        contamination=CONFIG["contamination"],
        n_estimators=CONFIG["n_estimators"],
    )

    # Treina com o dataset completo.
    # ← INTEGRAÇÃO REAL: considere treinar com dados históricos normais
    #   (ex: primeiros 30 dias sem eventos conhecidos) e aplicar predict()
    #   em dados novos para detecção em produção.
    detector.fit(df)
    df_resultado = detector.predict(df)

    # Avaliação simples: comparar detecções com ground truth do mock
    if "anomalia_real" in df_resultado.columns:
        reais     = df_resultado["anomalia_real"] != "normal"
        detectados = df_resultado["is_anomaly"] == -1
        vp = (reais & detectados).sum()
        fp = (~reais & detectados).sum()
        fn = (reais & ~detectados).sum()
        precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
        recall   = vp / (vp + fn) if (vp + fn) > 0 else 0
        print(f"\n  [Avaliação mock] VP={vp} | FP={fp} | FN={fn} | "
              f"Precisão={precisao:.1%} | Recall={recall:.1%}")

    sumarios = detector.get_anomaly_summary(df_resultado)
    if not sumarios:
        print("\n  Nenhuma anomalia detectada. Encerrando pipeline.")
        return

    # ── ETAPA 3: Geração de Laudos (RAG) ─────────────────────────────────────
    print("\n[ETAPA 3/3] Geração de pré-laudos técnicos (RAG)...")

    rag = RAGPipeline(
        pasta_manuais=CONFIG["pasta_manuais"],
        pasta_saida=CONFIG["pasta_laudos"],
        usar_mock_llm=CONFIG["usar_mock_llm"],
    )
    rag.carregar_base_conhecimento()

    # Gera laudos para os primeiros N eventos (demo) — remova o slice em produção
    laudos = rag.processar_anomalias(sumarios[:CONFIG["max_laudos_demo"]])

    # ── Sumário Final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  PIPELINE CONCLUÍDO")
    print(f"  • Eventos detectados: {len(sumarios)}")
    print(f"  • Laudos gerados: {len(laudos)}")
    print(f"  • Destino: {CONFIG['pasta_laudos']}/")
    print("=" * 65)

    if laudos:
        print("\n── EXEMPLO: Primeiro laudo gerado ─────────────────────────\n")
        print(laudos[0])


if __name__ == "__main__":
    main()
