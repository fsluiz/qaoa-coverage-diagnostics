# Controle no IBM Kingston: 13 blocos separados

- Job: `dan62g5r85ps73fdlp00`
- Backend: `ibm_kingston`
- Shots por bloco: 10,000
- Execuções de circuitos no total: 130,000
- Ângulos preservados: $\gamma=-0.04$ e $\beta=0.35$.
- Mitigação/configuração aplicada: DD XY4 e twirling de portas e medidas.

## Resultado principal

O produto das probabilidades ótimas locais foi $0.0014306274$. Isso corresponde a **143.06 hits ótimos por 100.000** (intervalo de 95%: 134.68 a 151.78).

Referências: circuito ideal = 182.54/100.000; circuito físico conjunto de 24 qubits = 79/100.000.

A execução separada reteve 78.4% da probabilidade ideal e foi 1.81 vezes a probabilidade observada no circuito conjunto.

Ela recuperou 61.9% da lacuna entre o circuito ideal e o circuito físico conjunto. Ainda permaneceu 21.6% abaixo do ideal.

## Resultado por bloco

| bloco | qubits originais | hits/shots | $P_i$ medido | $P_i$ ideal | razão | TV ideal |
|---|---:|---:|---:|---:|---:|---:|
| block_01 | [0, 1] | 9416/10000 | 0.941600 | 0.958540 | 0.982 | 0.017 |
| block_02 | [2, 3, 4, 5] | 5752/10000 | 0.575200 | 0.616986 | 0.932 | 0.042 |
| block_03 | [6, 7, 9] | 7177/10000 | 0.717700 | 0.750599 | 0.956 | 0.033 |
| block_04 | [8] | 5149/10000 | 0.514900 | 0.512881 | 1.004 | 0.002 |
| block_05 | [10] | 5130/10000 | 0.513000 | 0.512881 | 1.000 | 0.000 |
| block_06 | [11] | 5041/10000 | 0.504100 | 0.512881 | 0.983 | 0.009 |
| block_07 | [12, 13, 14] | 7398/10000 | 0.739800 | 0.778308 | 0.951 | 0.039 |
| block_08 | [15, 16, 17] | 3256/10000 | 0.325600 | 0.329536 | 0.988 | 0.016 |
| block_09 | [18] | 5059/10000 | 0.505900 | 0.512881 | 0.986 | 0.007 |
| block_10 | [19] | 7193/10000 | 0.719300 | 0.721908 | 0.996 | 0.003 |
| block_11 | [20] | 6124/10000 | 0.612400 | 0.613471 | 0.998 | 0.001 |
| block_12 | [21, 22] | 8340/10000 | 0.834000 | 0.852845 | 0.978 | 0.021 |
| block_13 | [23] | 6174/10000 | 0.617400 | 0.613471 | 1.006 | 0.004 |

## Realização física

- Tempo quântico informado pela IBM: 44.131 s.
- Portas físicas de dois qubits somadas nos 13 circuitos: 51.
- Maior profundidade física individual: 63.
- Qubits físicos distintos reutilizados entre os circuitos: 7 ([88, 89, 90, 122, 123, 124, 136]).
- Produto do proxy independente porta × leitura: 0.8220; ele corresponderia a 150.05 hits por 100.000.

| bloco | qubits físicos | 2Q físicas | profundidade | proxy porta×leitura |
|---|---:|---:|---:|---:|
| block_01 | [88, 89] | 3 | 16 | 0.9883 |
| block_02 | [122, 123, 124, 136] | 18 | 63 | 0.9452 |
| block_03 | [88, 89, 90] | 9 | 36 | 0.9670 |
| block_04 | [88] | 0 | 6 | 0.9978 |
| block_05 | [88] | 0 | 6 | 0.9978 |
| block_06 | [88] | 0 | 6 | 0.9978 |
| block_07 | [88, 89, 90] | 9 | 38 | 0.9670 |
| block_08 | [88, 89, 90] | 9 | 37 | 0.9670 |
| block_09 | [88] | 0 | 6 | 0.9978 |
| block_10 | [88] | 0 | 6 | 0.9978 |
| block_11 | [88] | 0 | 6 | 0.9978 |
| block_12 | [88, 89] | 3 | 15 | 0.9883 |
| block_13 | [88] | 0 | 6 | 0.9978 |

## Interpretação

Este número é uma reconstrução estatística. Os 13 blocos foram executados como circuitos independentes e o sucesso global foi calculado por

$$
P_{\mathrm{sep}}(\mathrm{ótimo})=\prod_{i=1}^{13}P_i(O_i).
$$

A igualdade usa a fatoração exata das 288 soluções ótimas em conjuntos locais. Ela não afirma que houve uma única medição simultânea dos 24 qubits. Se o produto se aproximar do ideal e superar claramente 79/100.000, a perda do circuito conjunto é compatível com o custo físico de mapear/executar todos os blocos de uma vez (mais qubits lidos, roteamento e ruído acumulado). Se o produto também ficar baixo, parte relevante da perda já ocorre dentro dos pequenos blocos.

Neste job, o resultado separado ficou entre o circuito conjunto e o ideal. Portanto, há perdas mensuráveis dentro dos blocos, mas elas não explicam toda a queda para 79/100.000. O circuito conjunto tinha 49 portas físicas de dois qubits e profundidade 64; os blocos somaram 51 portas, e o maior deles teve profundidade 63. A melhora não pode ser atribuída simplesmente a menos portas. Ela é consistente com reutilizar poucos qubits físicos bem calibrados e evitar execução simultânea/crosstalk entre os blocos.

Essa comparação não isola uma causa única: os dois jobs ocorreram em datas diferentes e receberam mapeamentos físicos diferentes. Assim, o resultado é evidência de uma penalidade da realização física conjunta, não uma medição exclusiva de crosstalk.
