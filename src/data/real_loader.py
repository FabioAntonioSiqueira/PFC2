"""
real_loader.py
==============
Conector com o banco PostgreSQL de origem (solarufg, servidor do LaMCAD).

As consultas ficam em sql/ e seguem o mapeamento validado no projeto PFC
(PFC/docs/dados/ANALISE_INVENTARIO.md, seções 2.7 e 2.10):

    Solis    → sql/extracao_solis.sql     (leituras_completas_solis)
    Solarman → sql/extracao_solarman.sql  (leituras_completas_solarman)
    WEG      → fora: só tem energia, sem tensão, corrente e temperatura.

PONTO DE SUBSTITUIÇÃO em main.py:
    # Antes (mock):
    df = generate_synthetic_data(...)

    # Depois (dados reais):
    from src.data.real_loader import load_canonical_dataframe
    df = load_canonical_dataframe(data_inicio="2025-12-01", data_fim="2026-06-01")

Pré-requisitos:
    pip install psycopg2-binary sqlalchemy python-dotenv
    Túnel SSH aberto (ver README do PFC): o banco responde em localhost:5433.

Configuração:
    Crie um arquivo `.env` na raiz do projeto (copie de .env.example).
    Nunca commite o .env no git (já está no .gitignore).

Segurança:
    O usuário do banco tem permissão de escrita. Por isso a sessão é aberta
    em modo somente leitura, com limite de tempo, e a transação nunca é
    confirmada (termina sempre em ROLLBACK).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

RAIZ = Path(__file__).resolve().parent.parent.parent
PASTA_SQL = RAIZ / "sql"

# Carrega variáveis de ambiente do arquivo .env na raiz do projeto
load_dotenv(RAIZ / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Conexão Somente Leitura
# ─────────────────────────────────────────────────────────────────────────────

TIMEOUT_CONSULTA_S = 300

def _env(nome_pg: str, nome_antigo: str, padrao: Optional[str] = None) -> Optional[str]:
    """Lê a variável no padrão libpq (PGHOST...) ou no nome antigo (DB_HOST...)."""
    return os.getenv(nome_pg) or os.getenv(nome_antigo) or padrao


def _get_engine():
    """
    Cria o engine SQLAlchemy com as credenciais do .env, em sessão somente leitura.

    Variáveis esperadas no .env (mesmas do projeto PFC):
        PGHOST     = localhost   (túnel SSH)
        PGPORT     = 5433        (porta local do túnel)
        PGDATABASE = solarufg
        PGUSER     = usuario
        PGPASSWORD = senha
    Os nomes antigos (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) ainda funcionam.
    """
    host     = _env("PGHOST", "DB_HOST", "localhost")
    port     = _env("PGPORT", "DB_PORT", "5433")
    dbname   = _env("PGDATABASE", "DB_NAME")
    user     = _env("PGUSER", "DB_USER")
    password = _env("PGPASSWORD", "DB_PASSWORD")

    if not all([dbname, user, password]):
        raise EnvironmentError(
            "Credenciais do banco não encontradas. "
            "Verifique o arquivo .env na raiz do projeto. "
            "Variáveis necessárias: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD"
        )

    url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=dbname,
    )
    opcoes = " ".join([
        "-c default_transaction_read_only=on",
        f"-c statement_timeout={TIMEOUT_CONSULTA_S * 1000}",
        "-c lock_timeout=2000",
        "-c idle_in_transaction_session_timeout=60000",
    ])
    return create_engine(
        url,
        connect_args={
            "options": opcoes,
            "application_name": "pfc2-inversores-readonly",
            "connect_timeout": 10,
        },
    )


def _executar_sql(engine, arquivo: str, params: dict) -> pd.DataFrame:
    """Executa um arquivo de sql/ e devolve o resultado. Nunca faz COMMIT."""
    sql = (PASTA_SQL / arquivo).read_text(encoding="utf-8")
    print(f"[real_loader] Executando sql/{arquivo}...")
    with engine.connect() as conn:
        try:
            df = pd.read_sql(text(sql), conn, params=params)
        finally:
            conn.rollback()
    print(f"[real_loader]   → {len(df):,} leituras brutas.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Conversão para o Schema Canônico
# ─────────────────────────────────────────────────────────────────────────────

# Sentinelas encontradas no inventário (ANALISE_INVENTARIO §2.5 e §2.10.7).
# Textos como 'nan', 'None' e '' já viram NaN em pd.to_numeric(errors="coerce").
SENTINELAS_TEMPERATURA = [-100.0, -99.6]

# Faixas físicas plausíveis; fora delas o valor vira NaN (não apaga a linha).
FAIXAS = {
    "potencia_kw":   (0.0, 1000.0),
    "tensao_v":      (0.0, 1000.0),
    "corrente_a":    (0.0, 1000.0),
    "temperatura_c": (-10.0, 120.0),
}


def _num(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def _media_fases(df: pd.DataFrame, colunas: list[str], ignorar_zero: bool) -> pd.Series:
    """Média das três fases, ignorando fases ausentes (e zeradas, se pedido)."""
    valores = df[colunas].apply(_num)
    if ignorar_zero:
        valores = valores.where(valores > 0)
    return valores.mean(axis=1, skipna=True)


def _tensao_e_corrente(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Tensão e corrente AC no formato do contrato canônico.

    As duas fontes medem tensão fase-neutro (~220 V, rede 220/380 V).
    O contrato (e as heurísticas do detector) usa tensão de linha (~380 V),
    então a média fase-neutro é multiplicada por √3 (rede equilibrada).
    A corrente é a média das três fases.

    Tensão 0 numa fase significa fase não medida (ex: Standby do Solarman,
    que só traz uma fase); corrente 0 é medição válida (sem geração).
    """
    tensao   = _media_fases(df, ["uac1", "uac2", "uac3"], ignorar_zero=True) * np.sqrt(3)
    corrente = _media_fases(df, ["iac1", "iac2", "iac3"], ignorar_zero=False)
    return tensao, corrente


def _canonico_solis(bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Regras do Solis (ANALISE_INVENTARIO §2.5, §2.10.3 e §2.10.5):
        - instante = timestr − 8 h (UTC), convertido para o horário de Brasília;
        - pac × pacpec dá o valor na unidade de pacstr (kW em 99,9%, W no resto).
    """
    utc = pd.to_datetime(bruto["timestr_utc8"], errors="coerce") - pd.Timedelta(hours=8)
    local = (
        utc.dt.tz_localize("UTC")
        .dt.tz_convert("America/Sao_Paulo")
        .dt.tz_localize(None)
    )

    valor_pac = _num(bruto["pac"]) * _num(bruto["pacpec"]).fillna(0.001)
    potencia_kw = np.where(
        bruto["pacstr"].str.strip().str.upper() == "W",
        valor_pac / 1000.0,
        valor_pac,
    )

    tensao, corrente = _tensao_e_corrente(bruto)
    return pd.DataFrame({
        "timestamp":       local,
        "inversor_id":     bruto["inversor_id"].astype(str),
        "potencia_kw":     potencia_kw,
        "energia_dia_kwh": _num(bruto["energia_dia_kwh"]),
        "tensao_v":        tensao,
        "corrente_a":      corrente,
        "temperatura_c":   _num(bruto["temperatura_c"]),
    })


def _canonico_solarman(bruto: pd.DataFrame) -> pd.DataFrame:
    """
    Regras do Solarman (ANALISE_INVENTARIO §2.10.4 e §2.10.7):
        - collecttime já está no horário de Brasília;
        - potência em W; temperatura do dissipador com sentinela -100 (Standby).
    """
    tensao, corrente = _tensao_e_corrente(bruto)
    temperatura = _num(bruto["temperatura_c"])
    em_standby = bruto["status"].astype(str).str.strip().str.lower() == "standby"
    temperatura = temperatura.mask(em_standby)

    return pd.DataFrame({
        "timestamp":       pd.to_datetime(bruto["timestamp_local"], errors="coerce"),
        "inversor_id":     bruto["inversor_id"].astype(str),
        "potencia_kw":     _num(bruto["potencia_w"]) / 1000.0,
        "energia_dia_kwh": _num(bruto["energia_dia_kwh"]),
        "tensao_v":        tensao,
        "corrente_a":      corrente,
        "temperatura_c":   temperatura,
    })


def _limpar_valores_invalidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Troca sentinelas e valores fisicamente impossíveis por NaN.

    Segue a regra do PFC: dado inválido não apaga a leitura inteira,
    só a grandeza afetada.
    """
    df = df.copy()
    df = df.dropna(subset=["timestamp"])

    df["temperatura_c"] = df["temperatura_c"].replace(SENTINELAS_TEMPERATURA, np.nan)
    for coluna, (minimo, maximo) in FAIXAS.items():
        df[coluna] = df[coluna].where(df[coluna].between(minimo, maximo))

    return df


def _energia_por_leitura(df: pd.DataFrame) -> pd.Series:
    """
    Converte a energia acumulada no dia em energia de cada leitura [kWh].

    A diferença entre leituras consecutivas do mesmo inversor e do mesmo dia
    dá a energia do intervalo. Quedas (reinício do contador) viram zero.
    Leituras sem energia ficam com 0 e não quebram a diferença das vizinhas.
    """
    validas = df.dropna(subset=["energia_dia_kwh"]).sort_values(["inversor_id", "timestamp"])
    dia = validas["timestamp"].dt.date
    delta = validas.groupby([validas["inversor_id"], dia])["energia_dia_kwh"].diff()
    # Primeira leitura do dia: o acumulado começa em zero à meia-noite
    delta = delta.fillna(validas["energia_dia_kwh"])
    return delta.clip(lower=0).reindex(df.index).fillna(0.0)


def _agregar_janelas_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega as leituras (~5 min) em janelas de 15 minutos por inversor.

    Janelas que ficarem sem alguma grandeza são descartadas, porque o
    Isolation Forest não aceita valores ausentes.
    """
    df = df.copy()
    df["energia_kwh"] = _energia_por_leitura(df)
    df = df.set_index("timestamp")

    agregado = (
        df.groupby("inversor_id")
        .resample("15min")
        .agg(
            potencia_kw=("potencia_kw",     "mean"),
            energia_kwh=("energia_kwh",     "sum"),
            tensao_v=("tensao_v",           "mean"),
            corrente_a=("corrente_a",       "mean"),
            temperatura_c=("temperatura_c", "mean"),
            n_leituras=("potencia_kw",      "size"),
        )
        .reset_index()
    )

    # Janelas criadas pelo resample sem nenhuma leitura (noite, lacunas)
    agregado = agregado[agregado["n_leituras"] > 0].drop(columns="n_leituras")

    cols_numericas = ["potencia_kw", "energia_kwh", "tensao_v", "corrente_a", "temperatura_c"]
    incompletas = agregado[cols_numericas].isna().any(axis=1)
    if incompletas.any():
        print(
            f"[real_loader]   → {incompletas.sum():,} janelas descartadas por "
            f"grandeza ausente ({incompletas.mean():.1%})."
        )
    return agregado[~incompletas]


# ─────────────────────────────────────────────────────────────────────────────
# Interface Pública
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical_dataframe(
    data_inicio: Optional[str] = None,
    data_fim: Optional[str] = None,
    incluir_solis: bool = True,
    incluir_solarman: bool = True,
) -> pd.DataFrame:
    """
    Carrega e retorna o DataFrame canônico a partir do PostgreSQL.

    Combina dados das plataformas Solis e Solarman em um único DataFrame
    no schema canônico, com janelas de 15 minutos. É um drop-in replacement
    do generate_synthetic_data() em main.py.

    Janela recomendada (ANALISE_INVENTARIO §2.10.7): dez/2025 a mai/2026,
    quando 9 a 10 inversores Solarman estão completos. O Solis cobre de
    out/2022 a ago/2026.

    Args:
        data_inicio: Data de início 'YYYY-MM-DD', horário de Brasília (opcional).
        data_fim:    Data de fim 'YYYY-MM-DD', exclusiva (opcional).
        incluir_solis:    Se True, inclui dados da plataforma Solis.
        incluir_solarman: Se True, inclui dados da plataforma Solarman.

    Returns:
        DataFrame no schema canônico com colunas:
            timestamp, inversor_id, potencia_kw, energia_kwh,
            tensao_v, corrente_a, temperatura_c
        inversor_id é a sigla do cadastro (ex: 'ODONTO-01', 'IF-INV-06').
    """
    if not (incluir_solis or incluir_solarman):
        raise ValueError("Nenhuma plataforma habilitada. Defina incluir_solis=True ou incluir_solarman=True.")

    params = {
        "data_inicio": data_inicio or "1900-01-01",
        "data_fim":    data_fim or "2100-01-01",
    }
    engine = _get_engine()
    frames: list[pd.DataFrame] = []

    if incluir_solis:
        bruto = _executar_sql(engine, "extracao_solis.sql", params)
        frames.append(_canonico_solis(bruto))

    if incluir_solarman:
        bruto = _executar_sql(engine, "extracao_solarman.sql", params)
        frames.append(_canonico_solarman(bruto))

    df_combined = pd.concat(frames, ignore_index=True)

    # Pipeline de qualidade de dados
    df_combined = _limpar_valores_invalidos(df_combined)
    df_combined = _agregar_janelas_15min(df_combined)

    df_combined = (
        df_combined
        .sort_values(["timestamp", "inversor_id"])
        .reset_index(drop=True)
    )

    if df_combined.empty:
        raise ValueError(
            f"Nenhuma janela válida entre {params['data_inicio']} e {params['data_fim']}. "
            "Confira o período (o Solarman só tem dados reais de abr/2025 a 03/06/2026)."
        )

    print(
        f"\n[real_loader] DataFrame canônico pronto: {len(df_combined):,} registros | "
        f"{df_combined['inversor_id'].nunique()} inversores\n"
        f"[real_loader] Período: {df_combined['timestamp'].min()} → {df_combined['timestamp'].max()}"
    )
    return df_combined
