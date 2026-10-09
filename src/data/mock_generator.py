"""
mock_generator.py
=================
Gerador de dados sintéticos para o pipeline de monitoramento de inversores
fotovoltaicos, seguindo o contrato de interface da Camada Canônica.

Contrato de Interface (Schema Canônico):
    timestamp     (datetime64): Início da janela de 15 minutos
    inversor_id   (object):     Identificador do inversor (ex: 'INV01')
    potencia_kw   (float64):    Potência média na janela [kW]
    energia_kwh   (float64):    Energia acumulada na janela [kWh]
    tensao_v      (float64):    Tensão média [V]
    corrente_a    (float64):    Corrente média [A]
    temperatura_c (float64):    Temperatura média [°C]

PONTO DE SUBSTITUIÇÃO (integração com dados reais):
    Ao receber o DataFrame real da Camada Canônica (Matheus),
    substitua a chamada `generate_synthetic_data()` em main.py por
    uma função que leia o DataFrame do PostgreSQL ou CSV.
    O schema de saída deve ser idêntico ao contrato acima.
    A coluna 'anomalia_real' não existirá nos dados reais — ela
    existe aqui apenas como ground truth para avaliação do detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tipos de Anomalia Suportados
# ─────────────────────────────────────────────────────────────────────────────

TIPOS_ANOMALIA = {
    "queda_potencia",     # Queda brusca de potência (< 15% do esperado)
    "sobretensao",        # Tensão CA acima de 120% do nominal
    "superaquecimento",   # Temperatura interna > 70 °C
    "falha_total",        # Potência, corrente e energia zeradas durante o dia
}


@dataclass
class AnomalyEvent:
    """
    Define um evento de anomalia a ser injetado nos dados sintéticos.

    Attributes:
        tipo:             Tipo da anomalia (ver TIPOS_ANOMALIA).
        inversor_id:      ID do inversor afetado (ex: 'INV02').
        dia:              Dia do período a ser afetado (0-indexed).
        hora_inicio:      Hora de início da anomalia (0–23).
        duracao_janelas:  Número de janelas de 15 min afetadas.
    """
    tipo: str
    inversor_id: str
    dia: int
    hora_inicio: int
    duracao_janelas: int

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS_ANOMALIA:
            raise ValueError(
                f"Tipo de anomalia '{self.tipo}' inválido. "
                f"Escolha entre: {TIPOS_ANOMALIA}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Perfil Solar (Irradiância Simulada)
# ─────────────────────────────────────────────────────────────────────────────

def _perfil_solar(minuto_do_dia: np.ndarray, potencia_pico_kw: float = 50.0) -> np.ndarray:
    """
    Calcula a potência esperada com base em um perfil solar gaussiano.

    Simula a curva de irradiância solar para Goiânia/GO:
        Nascer do sol : ~06:00
        Pôr do sol    : ~18:00
        Pico solar    : ~12:00

    Args:
        minuto_do_dia:    Array de minutos do dia (0–1439).
        potencia_pico_kw: Potência de pico do inversor [kW].

    Returns:
        Array de potências esperadas [kW], zero fora do horário solar.
    """
    NASCER_SOL = 6 * 60    # 360 min
    POR_DO_SOL = 18 * 60   # 1080 min
    PICO_SOLAR = (NASCER_SOL + POR_DO_SOL) / 2  # 720 min
    SIGMA = (POR_DO_SOL - NASCER_SOL) / 5.0     # "abertura" da gaussiana

    return np.where(
        (minuto_do_dia >= NASCER_SOL) & (minuto_do_dia <= POR_DO_SOL),
        potencia_pico_kw * np.exp(-0.5 * ((minuto_do_dia - PICO_SOLAR) / SIGMA) ** 2),
        0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Geração de Dados Normais por Inversor
# ─────────────────────────────────────────────────────────────────────────────

def _gerar_dados_normais(
    timestamps: pd.DatetimeIndex,
    inversor_id: str,
    potencia_pico_kw: float,
    tensao_nominal_v: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Gera dados de operação normal para um único inversor.

    Relações físicas modeladas:
        energia_kwh   = potencia_kw × (15/60)          [janela de 15 min]
        corrente_a    ≈ potencia_kw × 1000 / tensao_v  [lei de Ohm simplificada]
        temperatura_c = 25 + k × (potencia/pico) + ruído  [modelo térmico linear]

    Args:
        timestamps:       Índice de datas/horas das janelas.
        inversor_id:      Identificador do inversor.
        potencia_pico_kw: Potência de pico do inversor.
        tensao_nominal_v: Tensão nominal de operação.
        rng:              Gerador de números aleatórios para reprodutibilidade.

    Returns:
        DataFrame no schema canônico com coluna extra 'anomalia_real' = 'normal'.
    """
    minutos_do_dia = (timestamps.hour * 60 + timestamps.minute).values
    potencia_base  = _perfil_solar(minutos_do_dia, potencia_pico_kw)

    # Ruído gaussiano e sombreamento parcial ocasional (5% das janelas)
    ruido         = rng.normal(0, 0.8, len(timestamps))
    fator_nuvens  = np.where(
        rng.random(len(timestamps)) < 0.05,
        rng.uniform(0.3, 0.7, len(timestamps)),
        1.0,
    )
    potencia_kw   = np.maximum(potencia_base * fator_nuvens + ruido, 0.0)

    tensao_v      = tensao_nominal_v + rng.normal(0, 3.0, len(timestamps))
    corrente_a    = np.where(tensao_v > 0, (potencia_kw * 1000) / tensao_v, 0.0)
    temperatura_c = 25.0 + (potencia_kw / potencia_pico_kw) * 20.0 + rng.normal(0, 1.5, len(timestamps))
    energia_kwh   = potencia_kw * (15 / 60)

    return pd.DataFrame({
        "timestamp":     timestamps,
        "inversor_id":   inversor_id,
        "potencia_kw":   potencia_kw.round(3),
        "energia_kwh":   energia_kwh.round(4),
        "tensao_v":      tensao_v.round(2),
        "corrente_a":    corrente_a.round(3),
        "temperatura_c": temperatura_c.round(2),
        "anomalia_real": "normal",          # ground truth — não existe nos dados reais
    })


# ─────────────────────────────────────────────────────────────────────────────
# Injeção de Anomalias
# ─────────────────────────────────────────────────────────────────────────────

def _injetar_anomalia(df: pd.DataFrame, evento: AnomalyEvent) -> pd.DataFrame:
    """
    Injeta um evento de anomalia nas janelas especificadas de um inversor.

    Comportamento por tipo:
        queda_potencia:  potencia_kw cai para 5–15% do valor medido.
        sobretensao:     tensao_v sobe para 120–130% do nominal.
        superaquecimento: temperatura_c sobe para 70–85 °C.
        falha_total:     potencia, corrente e energia zeradas; temperatura normaliza.

    Args:
        df:     DataFrame do inversor (operação normal já gerada).
        evento: Configuração da anomalia a injetar.

    Returns:
        DataFrame com a anomalia injetada e 'anomalia_real' atualizado.
    """
    df = df.copy()

    data_base     = pd.Timestamp(df["timestamp"].min().date())
    hora_inicio   = data_base + pd.Timedelta(days=evento.dia, hours=evento.hora_inicio)
    hora_fim      = hora_inicio + pd.Timedelta(minutes=15 * evento.duracao_janelas)

    # Afeta somente janelas diurnas do inversor alvo
    mascara = (
        (df["inversor_id"] == evento.inversor_id)
        & (df["timestamp"] >= hora_inicio)
        & (df["timestamp"] < hora_fim)
        & (df["potencia_kw"] > 0.5)          # só durante geração ativa
    )

    if not mascara.any():
        print(
            f"  [mock] Aviso: nenhuma janela diurna encontrada para "
            f"{evento.inversor_id} no dia {evento.dia} às {evento.hora_inicio}h."
        )
        return df

    if evento.tipo == "queda_potencia":
        fator = float(np.random.uniform(0.05, 0.15))
        df.loc[mascara, "potencia_kw"]  *= fator
        df.loc[mascara, "energia_kwh"]  *= fator
        df.loc[mascara, "corrente_a"]   *= fator

    elif evento.tipo == "sobretensao":
        fator = float(np.random.uniform(1.20, 1.30))
        df.loc[mascara, "tensao_v"] *= fator

    elif evento.tipo == "superaquecimento":
        df.loc[mascara, "temperatura_c"] = float(np.random.uniform(70, 85))

    elif evento.tipo == "falha_total":
        df.loc[mascara, ["potencia_kw", "energia_kwh", "corrente_a"]] = 0.0
        df.loc[mascara, "temperatura_c"] = 25.0   # esfria sem carga

    df.loc[mascara, "anomalia_real"] = evento.tipo
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Interface Pública
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_inversores: int = 5,
    n_dias: int = 7,
    seed: int = 42,
    anomalias: Optional[list[AnomalyEvent]] = None,
) -> pd.DataFrame:
    """
    Gera o DataFrame sintético no schema canônico do contrato de interface.

    PONTO DE SUBSTITUIÇÃO — Integração com dados reais:
        Substitua esta chamada em main.py por uma função que leia o DataFrame
        canônico real do PostgreSQL/CSV produzido pelo Matheus.
        O schema de saída deve ser idêntico ao contrato de interface.
        A coluna 'anomalia_real' não existirá nos dados reais.

    Args:
        n_inversores: Número de inversores a simular.
        n_dias:       Número de dias de operação a simular (96 janelas/dia).
        seed:         Semente para reprodutibilidade.
        anomalias:    Lista de AnomalyEvent a injetar.
                      Se None, usa um conjunto padrão representativo.

    Returns:
        DataFrame com (n_inversores × n_dias × 96) registros.
        Colunas: timestamp, inversor_id, potencia_kw, energia_kwh,
                 tensao_v, corrente_a, temperatura_c, anomalia_real.
    """
    rng = np.random.default_rng(seed)

    # Configura cada inversor com potência de pico e tensão levemente diferentes
    configs = {
        f"INV{i+1:02d}": {
            "potencia_pico_kw": float(rng.uniform(45.0, 55.0)),
            "tensao_nominal_v": float(rng.uniform(375.0, 385.0)),
        }
        for i in range(n_inversores)
    }

    # Timestamps: n_dias completos em janelas de 15 minutos
    inicio     = pd.Timestamp("2026-01-05 00:00:00")    # segunda-feira
    timestamps = pd.date_range(start=inicio, periods=n_dias * 96, freq="15min")

    # Anomalias padrão se não fornecidas (uma de cada tipo, inversores e dias distintos)
    if anomalias is None:
        anomalias = [
            AnomalyEvent("queda_potencia",    "INV02", dia=1, hora_inicio=10, duracao_janelas=4),
            AnomalyEvent("sobretensao",        "INV04", dia=2, hora_inicio=13, duracao_janelas=3),
            AnomalyEvent("superaquecimento",   "INV01", dia=3, hora_inicio=11, duracao_janelas=6),
            AnomalyEvent("falha_total",        "INV03", dia=4, hora_inicio=9,  duracao_janelas=8),
            AnomalyEvent("queda_potencia",    "INV05", dia=5, hora_inicio=14, duracao_janelas=2),
        ]

    frames: list[pd.DataFrame] = []
    for inv_id, cfg in configs.items():
        df_inv = _gerar_dados_normais(
            timestamps=timestamps,
            inversor_id=inv_id,
            potencia_pico_kw=cfg["potencia_pico_kw"],
            tensao_nominal_v=cfg["tensao_nominal_v"],
            rng=rng,
        )
        for evento in anomalias:
            if evento.inversor_id == inv_id:
                df_inv = _injetar_anomalia(df_inv, evento)
        frames.append(df_inv)

    df_final = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp", "inversor_id"])
        .reset_index(drop=True)
    )

    n_anomalas = (df_final["anomalia_real"] != "normal").sum()
    print(
        f"[mock_generator] Dataset gerado: {len(df_final):,} registros | "
        f"{n_inversores} inversores | {n_dias} dias\n"
        f"[mock_generator] Anomalias injetadas: {n_anomalas} janelas "
        f"({n_anomalas / len(df_final) * 100:.1f}% do total)"
    )
    return df_final
