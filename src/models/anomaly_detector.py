"""
anomaly_detector.py
===================
Módulo de Detecção de Anomalias usando Isolation Forest (scikit-learn).

Estratégia:
    Um modelo Isolation Forest independente é treinado por inversor_id,
    capturando o perfil individual de cada equipamento.
    O pré-processamento (StandardScaler) também é por inversor,
    garantindo que diferenças de potência nominal entre inversores
    não sejam interpretadas como anomalias.

Saída:
    DataFrame enriquecido com 'is_anomaly' e 'anomaly_score' +
    lista de AnomaliaSumario prontos para entrada no módulo RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Colunas de features que alimentam o Isolation Forest.
# PONTO DE MELHORIA: ao integrar os dados reais, avalie adicionar features
# derivadas como potencia_por_irradiancia (performance ratio) se o sensor
# de irradiância estiver disponível na Camada Canônica.
FEATURE_COLS: list[str] = [
    "potencia_kw",
    "energia_kwh",
    "tensao_v",
    "corrente_a",
    "temperatura_c",
]

# Limiar de potência mínima para considerar uma janela como "diurna".
# Janelas noturnas têm valores naturalmente zerados e não são anomalias.
LIMIAR_POTENCIA_DIURNA: float = 0.5  # kW


# ─────────────────────────────────────────────────────────────────────────────
# Contrato de Saída para o Módulo RAG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomaliaSumario:
    """
    Sumário estruturado de um evento de anomalia detectado.

    Este é o contrato de interface entre o AnomalyDetector e o RAGPipeline.
    O módulo RAG consome esta estrutura para montar o contexto do prompt
    e gerar o pré-laudo técnico.

    Attributes:
        inversor_id:         ID do inversor afetado.
        timestamp_inicio:    ISO 8601 do início do evento.
        timestamp_fim:       ISO 8601 do fim do evento.
        n_janelas:           Número de janelas de 15 min afetadas.
        grandezas_afetadas:  Features com maior desvio no período.
        score_medio:         Score médio do Isolation Forest (mais negativo = mais anômalo).
        valores_medios:      Média de cada feature durante o evento.
        tipo_provavel:       Inferência heurística do tipo de falha.
    """
    inversor_id: str
    timestamp_inicio: str
    timestamp_fim: str
    n_janelas: int
    grandezas_afetadas: List[str]
    score_medio: float
    valores_medios: Dict[str, float]
    tipo_provavel: str


# ─────────────────────────────────────────────────────────────────────────────
# Detector Principal
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Detector de anomalias baseado em Isolation Forest, treinado por inversor.

    Uso típico:
        detector = AnomalyDetector(contamination=0.05)
        detector.fit(df_treino)
        df_resultado = detector.predict(df_novo)
        sumarios = detector.get_anomaly_summary(df_resultado)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            contamination: Proporção esperada de anomalias no conjunto de treino.
                           Ajuste este parâmetro nos dados reais com base no
                           histórico de falhas documentadas.
            n_estimators:  Número de árvores do Isolation Forest.
            random_state:  Semente para reprodutibilidade.
        """
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

        # Um modelo e um scaler por inversor_id
        self._models:  Dict[str, IsolationForest] = {}
        self._scalers: Dict[str, StandardScaler]  = {}
        self._is_fitted = False

    # ── Treinamento ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Treina um Isolation Forest independente para cada inversor_id.

        Treina apenas com janelas diurnas (potencia_kw > limiar) para evitar
        que o perfil noturno (zeros) distorça a distribuição aprendida.

        Args:
            df: DataFrame no schema canônico.

        Returns:
            self — permite encadeamento: detector.fit(df).predict(df).
        """
        df_diurno = df[df["potencia_kw"] > LIMIAR_POTENCIA_DIURNA].copy()

        if df_diurno.empty:
            raise ValueError(
                "Nenhuma janela diurna encontrada para treinamento. "
                f"Verifique se 'potencia_kw' tem valores > {LIMIAR_POTENCIA_DIURNA}."
            )

        inversores_treinados: list[str] = []
        for inv_id, grupo in df_diurno.groupby("inversor_id"):
            X = grupo[FEATURE_COLS].values

            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_scaled)

            self._scalers[str(inv_id)] = scaler
            self._models[str(inv_id)]  = model
            inversores_treinados.append(str(inv_id))

        self._is_fitted = True
        print(
            f"[AnomalyDetector] Treinamento concluído para "
            f"{len(inversores_treinados)} inversores: {inversores_treinados}"
        )
        return self

    # ── Predição ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica os modelos treinados e adiciona colunas de detecção ao DataFrame.

        Colunas adicionadas:
            is_anomaly    (int):   Convenção scikit-learn: 1 = normal, -1 = anomalia.
            anomaly_score (float): Score contínuo do IF (mais negativo → mais anômalo).

        Janelas noturnas (potencia_kw ≤ limiar) recebem is_anomaly = 1 e score = 0
        automaticamente, pois zeros noturnos são operação normal esperada.

        Args:
            df: DataFrame no schema canônico.

        Returns:
            Cópia do DataFrame com colunas 'is_anomaly' e 'anomaly_score'.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Chame fit() antes de predict()."
            )

        df = df.copy()
        df["is_anomaly"]    = 1      # default: normal
        df["anomaly_score"] = 0.0

        for inv_id, grupo in df.groupby("inversor_id"):
            inv_id_str = str(inv_id)
            if inv_id_str not in self._models:
                print(f"[AnomalyDetector] Aviso: inversor '{inv_id}' não visto no treino. Pulando.")
                continue

            # Aplica somente em janelas diurnas; noturnas já ficam como normal
            idx_diurno = grupo[grupo["potencia_kw"] > LIMIAR_POTENCIA_DIURNA].index
            if idx_diurno.empty:
                continue

            X        = df.loc[idx_diurno, FEATURE_COLS].values
            X_scaled = self._scalers[inv_id_str].transform(X)

            df.loc[idx_diurno, "is_anomaly"]    = self._models[inv_id_str].predict(X_scaled)
            df.loc[idx_diurno, "anomaly_score"] = self._models[inv_id_str].score_samples(X_scaled)

        n_anomalas = (df["is_anomaly"] == -1).sum()
        print(
            f"[AnomalyDetector] Detecção concluída: {n_anomalas} janelas anômalas "
            f"({n_anomalas / len(df) * 100:.1f}% do total)"
        )
        return df

    # ── Inferência do Tipo de Anomalia ───────────────────────────────────────

    @staticmethod
    def _inferir_tipo(grupo_anomalo: pd.DataFrame) -> str:
        """
        Heurística para inferir o tipo mais provável de falha.

        PONTO DE MELHORIA:
            Calibre os limiares abaixo com valores reais da usina
            após a integração com a Camada Canônica do Matheus.
            Uma abordagem mais robusta seria um classificador supervisionado
            treinado com eventos rotulados de manutenção histórica.

        Args:
            grupo_anomalo: Subset do DataFrame contendo apenas janelas anômalas
                           de um único bloco contíguo de um inversor.

        Returns:
            String com o tipo provável da anomalia.
        """
        media = grupo_anomalo[FEATURE_COLS].mean()

        if media["temperatura_c"] > 65:
            return "superaquecimento"
        if media["tensao_v"] > 420:
            return "sobretensao"
        if media["potencia_kw"] < 1.0 and media["temperatura_c"] < 30:
            return "falha_total"
        if media["potencia_kw"] < 10.0:
            return "queda_potencia"
        return "desvio_nao_classificado"

    # ── Extração do Sumário para o RAG ───────────────────────────────────────

    def get_anomaly_summary(self, df: pd.DataFrame) -> List[AnomaliaSumario]:
        """
        Consolida as janelas anômalas em eventos e retorna sumários estruturados.

        Agrupa janelas anômalas contíguas (gap ≤ 30 min) por inversor em
        "blocos de evento" e gera um AnomaliaSumario por bloco.

        O AnomaliaSumario é o contrato de entrada do módulo RAG.

        Args:
            df: DataFrame com colunas 'is_anomaly' e 'anomaly_score' calculadas.

        Returns:
            Lista de AnomaliaSumario ordenada por timestamp de início.
        """
        if "is_anomaly" not in df.columns:
            raise ValueError("Execute predict() antes de get_anomaly_summary().")

        df_anomalos = df[df["is_anomaly"] == -1].copy()
        if df_anomalos.empty:
            print("[AnomalyDetector] Nenhuma anomalia detectada para sumarizar.")
            return []

        sumarios: List[AnomaliaSumario] = []

        for inv_id, grupo in df_anomalos.groupby("inversor_id"):
            grupo = grupo.sort_values("timestamp").copy()

            # Identifica blocos contíguos: lacuna > 30 min separa eventos distintos
            grupo["bloco"] = (
                grupo["timestamp"].diff() > pd.Timedelta(minutes=30)
            ).cumsum()

            for _, bloco in grupo.groupby("bloco"):
                # Features com desvio acima da média — as "grandezas afetadas"
                desvios             = bloco[FEATURE_COLS].std()
                media_desvios       = desvios.mean()
                grandezas_afetadas  = desvios[desvios > media_desvios].index.tolist()
                if not grandezas_afetadas:
                    grandezas_afetadas = FEATURE_COLS

                sumario = AnomaliaSumario(
                    inversor_id=str(inv_id),
                    timestamp_inicio=bloco["timestamp"].min().isoformat(),
                    timestamp_fim=bloco["timestamp"].max().isoformat(),
                    n_janelas=len(bloco),
                    grandezas_afetadas=grandezas_afetadas,
                    score_medio=round(float(bloco["anomaly_score"].mean()), 4),
                    valores_medios={
                        col: round(float(bloco[col].mean()), 3)
                        for col in FEATURE_COLS
                    },
                    tipo_provavel=self._inferir_tipo(bloco),
                )
                sumarios.append(sumario)

        sumarios.sort(key=lambda s: s.timestamp_inicio)
        print(
            f"[AnomalyDetector] {len(sumarios)} evento(s) de anomalia consolidado(s)."
        )
        return sumarios
