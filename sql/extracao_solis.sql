-- Pergunta: Quais são as leituras do Solis (potência, energia do dia, tensão e corrente AC, temperatura) em um período, por inversor?
-- Motivo: Alimentar o DataFrame canônico do pipeline de detecção (src/data/real_loader.py). Mapeamento e regras vêm de PFC/docs/dados/ANALISE_INVENTARIO.md (seções 2.2.2, 2.7, 2.10.3 e 2.10.5).
-- Saída esperada: 1 linha por leitura (~5 min), colunas em texto; a conversão numérica, de unidade e de fuso é feita no loader.
-- Parâmetros: :data_inicio e :data_fim ('YYYY-MM-DD', horário de Brasília, fim exclusivo).
-- Tempo: timestr está sempre em UTC+8 (China), ou seja, horário de Brasília + 11 h. O filtro desloca o período em 11 h para comparar direto no texto.
SELECT
    s.timestr                                   AS timestr_utc8,
    s.inverter_identifier                       AS inversor_sn,
    COALESCE(i.inversor_sigla, s.inverter_identifier) AS inversor_id,
    s.pac,
    s.pacstr,
    s.pacpec,
    s.etoday                                    AS energia_dia_kwh,
    s.uac1, s.uac2, s.uac3,
    s.iac1, s.iac2, s.iac3,
    s.invertertemperature                       AS temperatura_c
FROM leituras_completas_solis s
LEFT JOIN inversor i ON i.inversor_sn = s.inverter_identifier
WHERE s.timestr >= to_char(CAST(:data_inicio AS timestamp) + interval '11 hours', 'YYYY-MM-DD HH24:MI:SS')
  AND s.timestr <  to_char(CAST(:data_fim    AS timestamp) + interval '11 hours', 'YYYY-MM-DD HH24:MI:SS')
ORDER BY s.inverter_identifier, s.timestr;
