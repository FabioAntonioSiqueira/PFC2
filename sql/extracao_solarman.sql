-- Pergunta: Quais são as leituras do Solarman (potência, energia do dia, tensão e corrente AC, temperatura) em um período, por inversor?
-- Motivo: Alimentar o DataFrame canônico do pipeline de detecção (src/data/real_loader.py). Mapeamento e regras vêm de PFC/docs/dados/ANALISE_INVENTARIO.md (seções 2.2.1, 2.7, 2.10.4 e 2.10.7).
-- Saída esperada: 1 linha por leitura (~5 a 6 min), colunas em texto; a conversão numérica e o tratamento de sentinelas são feitos no loader.
-- Parâmetros: :data_inicio e :data_fim ('YYYY-MM-DD', horário de Brasília, fim exclusivo).
-- Regras: collecttime já está no horário de Brasília; as linhas às 00:00:00 são placeholders vazios e ficam de fora, assim como o SN 'nan'.
SELECT
    s.collecttime                               AS timestamp_local,
    s.sn                                        AS inversor_sn,
    COALESCE(i.inversor_sigla, s.sn)            AS inversor_id,
    s."total ac output power (active)"          AS potencia_w,
    s."daily production (active)"               AS energia_dia_kwh,
    s."ac voltage r/u/a"                        AS uac1,
    s."ac voltage s/v/b"                        AS uac2,
    s."ac voltage t/w/c"                        AS uac3,
    s."ac current r/u/a"                        AS iac1,
    s."ac current s/v/b"                        AS iac2,
    s."ac current t/w/c"                        AS iac3,
    s."heat-sink temperature"                   AS temperatura_c,
    s."inverter status"                         AS status
FROM leituras_completas_solarman s
LEFT JOIN inversor i ON i.inversor_sn = s.sn
WHERE s.collecttime >= :data_inicio
  AND s.collecttime <  :data_fim
  AND s.collecttime NOT LIKE '% 00:00:00'
  AND s.sn <> 'nan'
ORDER BY s.sn, s.collecttime;
