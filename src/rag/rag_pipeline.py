"""
rag_pipeline.py
===============
Módulo RAG (Retrieval-Augmented Generation) para geração de pré-laudos
técnicos de anomalias detectadas em inversores fotovoltaicos.

Fluxo do pipeline:
    1. Ingere PDFs técnicos (manuais Solis/Solarman + normas ANEEL/ABNT)
    2. Divide em chunks e cria índice vetorial (FAISS + embeddings locais)
    3. Recebe AnomaliaSumario do AnomalyDetector
    4. Recupera os chunks mais relevantes por similaridade semântica
    5. Preenche o PromptTemplate com contexto + sumário
    6. Gera o pré-laudo via LLM (ou template mock se LLM não configurado)

PONTO DE SUBSTITUIÇÃO — LLM Real:
    Por padrão, `usar_mock_llm=True` gera laudos via template estruturado,
    sem necessidade de API Key. Para integrar um LLM real, veja o método
    `_gerar_com_llm()` e defina `usar_mock_llm=False` no construtor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Garante que o diretório raiz do projeto está no path para imports internos
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.anomaly_detector import AnomaliaSumario

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Base de Conhecimento Mock
# ─────────────────────────────────────────────────────────────────────────────
# Usada quando nenhum PDF real está disponível em data/manuais/.
# PONTO DE SUBSTITUIÇÃO: adicione os PDFs reais dos manuais Solis/Solarman
# e normas (ANEEL, ABNT NBR 16149, IEC 62109-1) na pasta data/manuais/.
# O pipeline os carregará e indexará automaticamente.

_CONHECIMENTO_MOCK = """
MANUAL TÉCNICO — INVERSOR SOLAR SOLIS (GINLONG TECHNOLOGIES)
Serie: S5-GR1P
=============================================================

CAPÍTULO 4 — DIAGNÓSTICO E RESOLUÇÃO DE FALHAS

4.1 Queda de Potência de Saída
- Sintoma: Potência de saída inferior a 15% da potência nominal durante
  período com irradiância superior a 300 W/m² (horário solar pleno).
- Causas possíveis: Degradação dos módulos FV, sombreamento parcial ou total,
  falha no rastreador MPPT, strings CC desconectadas, conector MC4 oxidado.
- Procedimento de diagnóstico:
  (1) Verificar tensão de entrada CC no display ou via app SolisCloud.
  (2) Medir corrente por string com alicate amperímetro no quadro CC.
  (3) Inspecionar visualmente conectores MC4 por oxidação ou folga.
  (4) Consultar log de eventos (menu: Diagnóstico > Histórico de Alarmes).
- Ação corretiva: Se tensão CC dentro do intervalo nominal e potência baixa,
  realizar inspeção termográfica das strings. Verificar parâmetros MPPT
  (Vmpp, Impp) no menu avançado. Limpar módulos se sujidade detectada.

4.2 Sobretensão na Rede (Código E007 — Grid Overvoltage)
- Sintoma: Tensão CA de saída superior a 110% da tensão nominal por mais
  de 3 ciclos consecutivos (normalmente > 253 V em redes 220 V).
- Causas possíveis: Regulação inadequada da concessionária, impedância
  elevada no ponto de conexão, cabo CA subdimensionado, transformador
  de distribuição sobrecarregado.
- Procedimento de diagnóstico:
  (1) Medir tensão no ponto de acoplamento comum (PAC) com multímetro
      calibrado durante o período de geração.
  (2) Verificar configuração do parâmetro de limite de sobretensão
      (menu: Config > Rede > Limite Vtensão AC).
  (3) Comparar com demais inversores da usina para isolar causa sistêmica.
- Ação corretiva: Notificar concessionária conforme ANEEL RN 1.000/2021,
  Art. 59 (qualidade de energia). Verificar bitola dos cabos CA. Reduzir
  potência injetada via parâmetro P12 se necessário. Registrar ocorrência.

4.3 Superaquecimento (Código E015 — Over Temperature)
- Sintoma: Temperatura interna do inversor superior a 70 °C. O equipamento
  pode entrar em modo de derate automático (redução de potência proporcional
  ao excesso de temperatura para proteção dos IGBTs).
- Causas possíveis: Ventilação insuficiente no abrigo, temperatura ambiente
  elevada (> 45 °C), obstrução das entradas/saídas de ar, falha ou desgaste
  do ventilador interno, acúmulo de pó nos dissipadores.
- Procedimento de diagnóstico:
  (1) Verificar temperatura ambiente com termômetro no abrigo.
  (2) Medir temperatura da carcaça com termômetro infravermelho.
  (3) Testar rotação do ventilador interno (audível ao ligar).
  (4) Inspecionar filtros de ar por obstrução.
- Ação corretiva: Garantir espaçamento mínimo de 300 mm em todas as faces.
  Limpar filtros de ar e dissipadores (ar comprimido). Se Tamb > 45 °C,
  avaliar climatização do abrigo. Substituir ventilador se travado.
  Consultar Tabela 4-3 do manual para curva de derate por temperatura.

4.4 Falha Total / Desconexão Completa (Códigos E001–E004)
- Sintoma: Ausência total de geração (potência, corrente, energia = 0)
  durante período com irradiância > 200 W/m². LED frontal vermelho fixo.
- Causas possíveis: Trip de proteção ativado (subtensão/sobretensão CC),
  fusível CC queimado, disjuntor CA desarmado, falha do relé de saída,
  desconexão do DPS (dispositivo de proteção contra surto).
- Procedimento de diagnóstico:
  (1) Verificar LEDs de status: vermelho fixo = falha crítica.
  (2) Ler código de erro no display (menu: Status > Código de Falha).
  (3) Verificar disjuntores CC e CA no quadro de distribuição.
  (4) Medir tensão CC na entrada do inversor (esperar 5 min após o sol).
  (5) Verificar fusíveis CC individuais de cada string.
- Ação corretiva:
  E001–E004: Verificar tensão CC (mín. 200 V, máx. 600 V para este modelo).
  E010–E019: Falha interna — acionar suporte técnico autorizado Solis.
  Sem código: verificar todas as chaves seccionadoras e fusíveis CC.

NORMAS E REGULAMENTAÇÕES APLICÁVEIS:
- ABNT NBR 16149:2013 — Conexão de micro e minigeração distribuída à rede
- ABNT NBR 16150:2013 — Conformidade para interface de conexão
- ANEEL RN 1.000/2021 — Condições gerais de fornecimento de energia elétrica
- IEC 62109-1:2010 — Segurança para conversores de potência fotovoltaicos
- IEC 61724-1:2021 — Monitoramento de desempenho de sistemas FV
- IEEE 1547-2018 — Interconexão de recursos distribuídos

MANUTENÇÃO PREVENTIVA RECOMENDADA:
- Mensal:  Verificação visual dos conectores, limpeza dos filtros de ar.
- Trimestral: Inspeção termográfica dos módulos e string boxes.
- Semestral:  Verificação de aperto das conexões CC e CA, limpeza interna.
- Anual:   Verificação dos DPS, troca dos filtros, calibração de sensores.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template do Laudo Técnico
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_LAUDO = PromptTemplate(
    input_variables=["contexto_tecnico", "sumario_anomalia"],
    template="""Você é um engenheiro especialista em sistemas fotovoltaicos e diagnóstico \
de inversores solares.

Com base no contexto técnico recuperado dos manuais e normas vigentes, e no sumário \
da anomalia detectada pelo sistema de monitoramento, redija um Pré-Laudo Técnico \
estruturado, objetivo e profissional.

======= CONTEXTO TÉCNICO (Manuais / Normas recuperados) =======
{contexto_tecnico}

======= SUMÁRIO DA ANOMALIA DETECTADA =======
{sumario_anomalia}

======= PRÉ-LAUDO TÉCNICO =======
Redija o laudo com as seguintes seções:

1. IDENTIFICAÇÃO DO EVENTO
   Equipamento, período, duração e grandezas afetadas.

2. DESCRIÇÃO DO COMPORTAMENTO OBSERVADO
   Descreva objetivamente o que os dados indicam, com os valores registrados.

3. HIPÓTESES DE CAUSA
   Liste as causas mais prováveis com base nos manuais técnicos recuperados.

4. RECOMENDAÇÕES DE AÇÃO
   Ações imediatas e de médio prazo, com referências normativas.

5. NÍVEL DE URGÊNCIA
   Classifique: CRÍTICO / ALTO / MÉDIO / BAIXO — com justificativa.

PRÉ-LAUDO:""",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline RAG
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Pipeline RAG para geração de pré-laudos técnicos de anomalias em inversores.

    Responsabilidades:
        - Indexar documentos técnicos (PDFs ou mock) em vector store FAISS.
        - Recuperar contexto relevante por busca semântica.
        - Formatar o prompt com contexto + sumário de anomalia.
        - Gerar o pré-laudo via LLM real ou template estruturado (mock).
        - Salvar laudos em disco em outputs/laudos/.
    """

    def __init__(
        self,
        pasta_manuais: str = "data/manuais",
        pasta_saida: str = "outputs/laudos",
        modelo_embedding: str = "sentence-transformers/all-MiniLM-L6-v2",
        usar_mock_llm: bool = True,
    ) -> None:
        """
        Args:
            pasta_manuais:    Diretório com os PDFs técnicos a indexar.
            pasta_saida:      Diretório onde os laudos serão salvos.
            modelo_embedding: Modelo HuggingFace para embeddings (roda localmente,
                              sem API Key — download único de ~90 MB).
            usar_mock_llm:    True = gera laudo por template estruturado (sem LLM).
                              False = usa LLM configurado em _gerar_com_llm().
                              PONTO DE SUBSTITUIÇÃO: defina False quando o LLM
                              estiver disponível e configurado.
        """
        self.pasta_manuais  = Path(pasta_manuais)
        self.pasta_saida    = Path(pasta_saida)
        self.usar_mock_llm  = usar_mock_llm

        self.pasta_saida.mkdir(parents=True, exist_ok=True)

        print("[RAGPipeline] Carregando modelo de embeddings (primeira vez pode demorar ~30s)...")
        self._embeddings = HuggingFaceEmbeddings(model_name=modelo_embedding)
        self._vector_store: Optional[FAISS] = None
        self._carregado = False

    # ── Indexação ─────────────────────────────────────────────────────────────

    def carregar_base_conhecimento(self) -> None:
        """
        Carrega e indexa os documentos técnicos no vector store FAISS.

        Se a pasta data/manuais/ contiver PDFs, os carrega e indexa.
        Caso contrário, usa a base de conhecimento mock embutida,
        permitindo testes completos do pipeline sem PDFs reais.

        PONTO DE SUBSTITUIÇÃO:
            Adicione os PDFs dos manuais Solis/Solarman e as normas
            (ANEEL RN 1.000, ABNT NBR 16149, IEC 62109-1) em data/manuais/.
        """
        tem_pdfs = (
            self.pasta_manuais.exists()
            and any(self.pasta_manuais.glob("*.pdf"))
        )

        if tem_pdfs:
            n_pdfs = len(list(self.pasta_manuais.glob("*.pdf")))
            print(f"[RAGPipeline] Carregando {n_pdfs} PDF(s) de '{self.pasta_manuais}'...")
            loader    = PyPDFDirectoryLoader(str(self.pasta_manuais))
            documentos = loader.load()
            print(f"[RAGPipeline] {len(documentos)} páginas carregadas.")
        else:
            print(
                f"[RAGPipeline] Nenhum PDF em '{self.pasta_manuais}'. "
                "Usando base de conhecimento mock."
            )
            documentos = [
                Document(
                    page_content=_CONHECIMENTO_MOCK,
                    metadata={"source": "mock_manual_solis", "page": 0},
                )
            ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(documentos)
        print(f"[RAGPipeline] {len(chunks)} chunks indexados no vector store FAISS.")

        self._vector_store = FAISS.from_documents(chunks, self._embeddings)
        self._carregado    = True

    # ── Recuperação de Contexto ───────────────────────────────────────────────

    def _recuperar_contexto(self, query: str, k: int = 4) -> str:
        """
        Busca os k chunks mais semanticamente similares à query no FAISS.

        Args:
            query: Texto descrevendo a anomalia para busca contextual.
            k:     Número de chunks a recuperar.

        Returns:
            Texto concatenado dos chunks mais relevantes.
        """
        if not self._carregado or self._vector_store is None:
            raise RuntimeError("Chame carregar_base_conhecimento() primeiro.")

        docs = self._vector_store.similarity_search(query, k=k)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # ── Formatação do Sumário ─────────────────────────────────────────────────

    @staticmethod
    def _formatar_sumario(sumario: AnomaliaSumario) -> str:
        """Converte AnomaliaSumario em texto legível para o prompt."""
        linhas_valores = "\n".join(
            f"  • {k.replace('_', ' ').title()}: {v}"
            for k, v in sumario.valores_medios.items()
        )
        return (
            f"Inversor       : {sumario.inversor_id}\n"
            f"Início         : {sumario.timestamp_inicio}\n"
            f"Fim            : {sumario.timestamp_fim}\n"
            f"Duração        : {sumario.n_janelas} janelas "
            f"({sumario.n_janelas * 15} minutos)\n"
            f"Tipo provável  : {sumario.tipo_provavel.upper().replace('_', ' ')}\n"
            f"Score IF médio : {sumario.score_medio} "
            f"(referência: próximo de -1 = muito anômalo)\n"
            f"Grandezas c/ maior desvio: {', '.join(sumario.grandezas_afetadas)}\n"
            f"Valores médios no período anômalo:\n{linhas_valores}"
        )

    # ── Geração do Laudo ──────────────────────────────────────────────────────

    def _gerar_com_mock_template(
        self,
        contexto: str,
        sumario: AnomaliaSumario,
    ) -> str:
        """
        Gera o pré-laudo usando template estruturado, sem LLM externo.

        Utilizado quando usar_mock_llm=True.
        Suficiente para validar o pipeline ponta-a-ponta sem API Key.
        """
        urgencia_map = {
            "falha_total":             ("CRÍTICO", "Geração completamente interrompida durante período solar."),
            "superaquecimento":        ("ALTO",    "Risco de dano permanente aos IGBTs e componentes internos."),
            "sobretensao":             ("ALTO",    "Risco à integridade da rede elétrica e do equipamento."),
            "queda_potencia":          ("MÉDIO",   "Perda de geração significativa sem falha total detectada."),
            "desvio_nao_classificado": ("BAIXO",   "Desvio estatístico detectado; causa a ser investigada."),
        }
        nivel, justificativa = urgencia_map.get(
            sumario.tipo_provavel,
            ("BAIXO", "Investigar manualmente.")
        )
        tipo_fmt = sumario.tipo_provavel.upper().replace("_", " ")
        v = sumario.valores_medios

        # Hipóteses com base no tipo inferido (extrato do contexto mock)
        hipoteses_map = {
            "queda_potencia": (
                "  a) Degradação ou sombreamento parcial de módulos FV.\n"
                "  b) Falha no rastreador MPPT (tensão Vmpp fora do intervalo).\n"
                "  c) Conector MC4 com mau contato ou oxidação.\n"
                "  d) String CC parcialmente desconectada."
            ),
            "sobretensao": (
                "  a) Regulação inadequada da concessionária distribuidora.\n"
                "  b) Impedância elevada no ponto de acoplamento comum (PAC).\n"
                "  c) Cabo CA subdimensionado gerando queda de tensão reversa."
            ),
            "superaquecimento": (
                "  a) Ventilação insuficiente no abrigo de instalação.\n"
                "  b) Temperatura ambiente superior a 45 °C no local.\n"
                "  c) Filtros de ar obstruídos ou ventilador interno com falha.\n"
                "  d) Acúmulo de pó nos dissipadores de calor."
            ),
            "falha_total": (
                "  a) Trip de proteção ativado (fusível CC queimado ou DPS atuado).\n"
                "  b) Disjuntor CA desarmado no quadro de distribuição.\n"
                "  c) Falha interna do relé de saída CA.\n"
                "  d) Tensão CC abaixo do mínimo operacional (< 200 V)."
            ),
        }
        hipoteses = hipoteses_map.get(
            sumario.tipo_provavel,
            "  a) Desvio estatístico detectado pelo Isolation Forest.\n"
            "  b) Investigar com inspeção física no equipamento."
        )

        return (
            "═══════════════════════════════════════════════════════════════\n"
            "              PRÉ-LAUDO TÉCNICO AUTOMÁTICO\n"
            "    Sistema de Diagnóstico Preditivo FV — PFC2 / EMC / UFG\n"
            "═══════════════════════════════════════════════════════════════\n\n"
            "1. IDENTIFICAÇÃO DO EVENTO\n"
            f"   Equipamento    : {sumario.inversor_id}\n"
            f"   Período        : {sumario.timestamp_inicio}  →  {sumario.timestamp_fim}\n"
            f"   Duração        : {sumario.n_janelas} janelas de 15 min "
            f"({sumario.n_janelas * 15} minutos)\n"
            f"   Grandezas      : {', '.join(sumario.grandezas_afetadas)}\n"
            f"   Tipo detectado : {tipo_fmt}\n\n"
            "2. COMPORTAMENTO OBSERVADO\n"
            f"   O inversor {sumario.inversor_id} apresentou desvio significativo\n"
            "   em relação ao comportamento típico do conjunto na mesma janela\n"
            "   temporal. Valores médios registrados no período anômalo:\n"
            f"     • Potência   : {v.get('potencia_kw', 'N/A')} kW\n"
            f"     • Energia    : {v.get('energia_kwh', 'N/A')} kWh\n"
            f"     • Tensão     : {v.get('tensao_v', 'N/A')} V\n"
            f"     • Corrente   : {v.get('corrente_a', 'N/A')} A\n"
            f"     • Temperatura: {v.get('temperatura_c', 'N/A')} °C\n"
            f"   Score Isolation Forest (médio): {sumario.score_medio}\n\n"
            "3. HIPÓTESES DE CAUSA\n"
            "   Com base nos manuais técnicos e normas indexadas:\n"
            f"{hipoteses}\n\n"
            "4. RECOMENDAÇÕES DE AÇÃO\n"
            "   Imediato:\n"
            f"     → Inspecionar fisicamente o {sumario.inversor_id}.\n"
            "     → Verificar LEDs de status e código de erro no display.\n"
            "     → Checar disjuntores CC e CA no quadro de distribuição.\n"
            "   Médio prazo:\n"
            "     → Realizar inspeção termográfica das strings e conectores.\n"
            "     → Verificar parâmetros MPPT no menu de configuração avançada.\n"
            "   Normas de referência:\n"
            "     → ABNT NBR 16149 | IEC 62109-1 | ANEEL RN 1.000/2021\n\n"
            "5. NÍVEL DE URGÊNCIA\n"
            f"   [{nivel}] — {justificativa}\n\n"
            "───────────────────────────────────────────────────────────────\n"
            "AVISO: Pré-laudo gerado automaticamente para orientação técnica.\n"
            "Validação obrigatória por engenheiro responsável habilitado.\n"
            "Modo: Template Mock | LLM: não configurado\n"
            "═══════════════════════════════════════════════════════════════\n"
        )

    def _gerar_com_llm(self, prompt_completo: str) -> str:
        """
        PONTO DE SUBSTITUIÇÃO — Integração com LLM Real.

        Descomente e configure um dos blocos abaixo conforme o LLM disponível.
        Defina usar_mock_llm=False no construtor para ativar este método.

        ── OPÇÃO A: OpenAI (GPT-4o) ────────────────────────────────────────
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        return llm.invoke(prompt_completo).content

        ── OPÇÃO B: Anthropic (Claude claude-sonnet-4-6) ──────────────────────────
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)
        return llm.invoke(prompt_completo).content

        ── OPÇÃO C: Local via Ollama (sem custo, sem internet) ─────────────
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3")
        return llm.invoke(prompt_completo)
        """
        raise NotImplementedError(
            "LLM real não configurado. Implemente _gerar_com_llm() "
            "ou use usar_mock_llm=True no construtor."
        )

    # ── Interface Principal ───────────────────────────────────────────────────

    def gerar_laudo(self, sumario: AnomaliaSumario) -> str:
        """
        Orquestra a geração de um pré-laudo para um evento de anomalia.

        Args:
            sumario: AnomaliaSumario produzido pelo AnomalyDetector.

        Returns:
            Texto completo do pré-laudo técnico.
        """
        if not self._carregado:
            raise RuntimeError("Chame carregar_base_conhecimento() antes de gerar laudos.")

        # Monta query de busca com base no tipo de anomalia
        query = (
            f"{sumario.tipo_provavel.replace('_', ' ')} "
            f"inversor solar diagnóstico causas procedimento manutenção"
        )
        contexto        = self._recuperar_contexto(query)
        sumario_texto   = self._formatar_sumario(sumario)
        prompt_completo = _PROMPT_LAUDO.format(
            contexto_tecnico=contexto,
            sumario_anomalia=sumario_texto,
        )

        if self.usar_mock_llm:
            laudo = self._gerar_com_mock_template(contexto, sumario)
        else:
            laudo = self._gerar_com_llm(prompt_completo)

        # Persiste o laudo em disco
        nome_arquivo = (
            f"{sumario.inversor_id}_{sumario.tipo_provavel}_"
            f"{sumario.timestamp_inicio[:10]}.txt"
        )
        caminho = self.pasta_saida / nome_arquivo
        caminho.write_text(laudo, encoding="utf-8")
        print(f"[RAGPipeline] Laudo salvo em: {caminho}")

        return laudo

    def processar_anomalias(self, sumarios: List[AnomaliaSumario]) -> List[str]:
        """
        Processa uma lista de sumários e gera um laudo para cada evento.

        Args:
            sumarios: Lista de AnomaliaSumario do AnomalyDetector.

        Returns:
            Lista de textos dos pré-laudos gerados.
        """
        if not sumarios:
            print("[RAGPipeline] Nenhuma anomalia para processar.")
            return []

        laudos: List[str] = []
        for i, sumario in enumerate(sumarios, 1):
            print(
                f"[RAGPipeline] Gerando laudo {i}/{len(sumarios)}: "
                f"{sumario.inversor_id} — {sumario.tipo_provavel}"
            )
            laudos.append(self.gerar_laudo(sumario))

        return laudos
