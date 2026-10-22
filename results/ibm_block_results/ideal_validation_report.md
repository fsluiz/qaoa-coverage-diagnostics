# Validação local ideal do controle por 13 blocos

- Job: `simulação local`
- Backend: `ibm_kingston`
- Shots por bloco: 10,000
- Execuções de circuitos no total: 130,000
- Ângulos preservados: $\gamma=-0.04$ e $\beta=0.35$.
- Configuração prevista para o hardware: DD XY4 e twirling de portas e medidas.

## Resultado principal

O produto das probabilidades ótimas locais foi $0.0019399019$. Isso corresponde a **193.99 hits ótimos por 100.000** (intervalo de 95%: 182.91 a 205.42).

Referências: circuito ideal = 182.54/100.000; circuito físico conjunto de 24 qubits = 79/100.000.

A execução separada reteve 106.3% da probabilidade ideal e foi 2.46 vezes a probabilidade observada no circuito conjunto.

Ela recuperou 111.1% da lacuna entre o circuito ideal e o circuito físico conjunto. Ainda permaneceu -6.3% abaixo do ideal.

## Resultado por bloco

| bloco | qubits originais | hits/shots | $P_i$ medido | $P_i$ ideal | razão | TV ideal |
|---|---:|---:|---:|---:|---:|---:|
| block_01 | [0, 1] | 9602/10000 | 0.960200 | 0.958540 | 1.002 | 0.006 |
| block_02 | [2, 3, 4, 5] | 6222/10000 | 0.622200 | 0.616986 | 1.008 | 0.011 |
| block_03 | [6, 7, 9] | 7579/10000 | 0.757900 | 0.750599 | 1.010 | 0.011 |
| block_04 | [8] | 5114/10000 | 0.511400 | 0.512881 | 0.997 | 0.001 |
| block_05 | [10] | 5155/10000 | 0.515500 | 0.512881 | 1.005 | 0.003 |
| block_06 | [11] | 5158/10000 | 0.515800 | 0.512881 | 1.006 | 0.003 |
| block_07 | [12, 13, 14] | 7807/10000 | 0.780700 | 0.778308 | 1.003 | 0.006 |
| block_08 | [15, 16, 17] | 3334/10000 | 0.333400 | 0.329536 | 1.012 | 0.009 |
| block_09 | [18] | 5187/10000 | 0.518700 | 0.512881 | 1.011 | 0.006 |
| block_10 | [19] | 7232/10000 | 0.723200 | 0.721908 | 1.002 | 0.001 |
| block_11 | [20] | 6167/10000 | 0.616700 | 0.613471 | 1.005 | 0.003 |
| block_12 | [21, 22] | 8547/10000 | 0.854700 | 0.852845 | 1.002 | 0.002 |
| block_13 | [23] | 6122/10000 | 0.612200 | 0.613471 | 0.998 | 0.001 |

## Interpretação

Este número é uma reconstrução estatística. Os 13 blocos foram executados como circuitos independentes e o sucesso global foi calculado por

$$
P_{\mathrm{sep}}(\mathrm{ótimo})=\prod_{i=1}^{13}P_i(O_i).
$$

A igualdade usa a fatoração exata das 288 soluções ótimas em conjuntos locais. Ela não afirma que houve uma única medição simultânea dos 24 qubits. Se o produto se aproximar do ideal e superar claramente 79/100.000, a perda do circuito conjunto é compatível com o custo físico de mapear/executar todos os blocos de uma vez (mais qubits lidos, roteamento e ruído acumulado). Se o produto também ficar baixo, parte relevante da perda já ocorre dentro dos pequenos blocos.

> Este arquivo contém somente amostragem multinomial da distribuição ideal. Ele valida o programa, mas não é um resultado do IBM Kingston.
