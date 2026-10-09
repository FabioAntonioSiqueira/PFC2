# Diagnóstico Preditivo em Inversores Fotovoltaicos
**PFC2 — EMC / UFG | Fábio Siqueira & Matheus Fernandes**  
Orientador: Prof. Marcelo Stehling de Castro

---

## Visão Geral

Sistema integrado de monitoramento e diagnóstico preditivo para inversores solares,
organizado em três módulos com desenvolvimento orientado por **contrato de interface**:

```
[Camada Canônica — Matheus]
  PostgreSQL (Solis + Solarman) → DataFrame canônico (15 min)
          ↓
[Detecção de Anomalias — Fábio]        src/models/anomaly_detector.py
  Isolation Forest por inversor → AnomaliaSumario[]
          ↓
[Geração de Laudos RAG — Fábio]        src/rag/rag_pipeline.py
  Busca vetorial (FAISS) + LLM → Pré-Laudo Técnico (.txt)
```

---

## Estrutura do Projeto

```
pfc2-inversores/
├── src/
│   ├── data/
│   │   └── mock_generator.py       # Gerador de dados sintéticos (mock)
│   ├── models/
│   │   └── anomaly_detector.py     # Isolation Forest + sumário de anomalias
│   └── rag/
│       └── rag_pipeline.py         # RAG: indexação de PDFs + geração de laudos
├── data/
│   └── manuais/                    # ← Coloque os PDFs técnicos aqui
├── outputs/
│   └── laudos/                     # Laudos gerados automaticamente
├── notebooks/                      # Jupyter notebooks exploratórios
├── main.py                         # Script de execução ponta-a-ponta
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Ambiente Python
```bash
# Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 2. Execute o pipeline completo
```bash
python main.py
```

Na primeira execução, o modelo de embeddings (~90 MB) será baixado automaticamente.

---

## Contrato de Interface (Schema Canônico)

| Coluna          | Tipo         | Descrição                          |
|-----------------|--------------|------------------------------------|
| `timestamp`     | datetime64   | Início da janela de 15 minutos     |
| `inversor_id`   | object       | ID do inversor (ex: `INV01`)       |
| `potencia_kw`   | float64      | Potência média na janela [kW]      |
| `energia_kwh`   | float64      | Energia acumulada na janela [kWh]  |
| `tensao_v`      | float64      | Tensão média [V]                   |
| `corrente_a`    | float64      | Corrente média [A]                 |
| `temperatura_c` | float64      | Temperatura média [°C]             |

---

## Integração com Dados Reais (Camada Canônica)

Quando o Matheus disponibilizar o DataFrame canônico, substitua em `main.py`:

```python
# ANTES (mock)
df = generate_synthetic_data(n_inversores=5, n_dias=7)

# DEPOIS (dados reais)
from src.data.real_loader import load_canonical_dataframe
df = load_canonical_dataframe(host="...", port=5432, db="inversores")
```

O restante do pipeline (anomaly_detector, rag_pipeline) **permanece inalterado**.

---

## Integração com LLM Real

Em `main.py`, defina `"usar_mock_llm": False` e implemente em `rag_pipeline.py`:

```python
# Opção A — OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
return llm.invoke(prompt_completo).content

# Opção B — Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)
return llm.invoke(prompt_completo).content

# Opção C — Local (Ollama, sem custo)
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
return llm.invoke(prompt_completo)
```

---

## Adicionando Manuais Técnicos

Coloque os PDFs na pasta `data/manuais/`. O pipeline os indexará automaticamente:

```
data/manuais/
├── manual_solis_s5.pdf
├── manual_solarman.pdf
├── abnt_nbr_16149.pdf
└── aneel_rn_1000_2021.pdf
```

---

## Normas de Referência

- ABNT NBR 16149:2013 — Conexão de micro e minigeração distribuída
- ABNT NBR 16150:2013 — Procedimentos de ensaio de conformidade
- ANEEL RN 1.000/2021 — Condições gerais de fornecimento
- IEC 62109-1:2010 — Segurança para conversores de potência FV
- IEC 61724-1:2021 — Monitoramento de desempenho de sistemas FV
- IEEE 1547-2018 — Interconexão de recursos distribuídos
