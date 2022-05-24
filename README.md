<h1 align="center"> Projeção de Vendas 📊 </h1>

<h3 align="center"> Projeto realizado durante a semana "Intensivão de Python" proporcionado pela Hashtag Treinamentos. </h3>

Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio. A base de dados encontra-se já no repositório para facilitar a visualização do mesmo.

##

<h3 align="left"> Para execução do processo, segui os seguintes passos: </h3>

1. Entendimento do Desafio;
2. Entendimento da Área/Empresa;
3. Extração/Obtenção de Dados;
4. Ajuste de Dados (Tratamento/Limpeza);
5. Análise Exploratória;
6. Modelagem + Algoritmos;
7. Interpretação de Resultados.

**Observações:**
- TV, Jornal e Rádio estão em milhares de reais;
- Vendas estão em milhões.

<h2 align="left"> Introdução </h2>

Após a importação das bibliotecas necessárias e da base de dados, foi feita uma análise dos dados através da função _info()_ que permite visualizarmos todas colunas da tabela e como estão esses dados. Também foi feito um gráfico _heatmap_ para observarmos a correlação entre as linhas e colunas.

<h2 align="left"> Modelo de Machine Learning </h2>

Com o objetivo de criar um modelo de previsão de vendas, foi utilizado um modelo de Regressão. Para isso, foram divididos os dados de treino e os dados de teste para posteriormente serem utilizados na criação do modelo. Aqui foram encontrados dois possíveis modelos:
- Random Forest (Árvore de Decisão);
- Regressão Linear.
Após os testes, é observado uma acurácia de ~96% para o modelo Random Forest, tornando-o a melhor opção para esse desafio.

<h2 align="left"> Conclusão </h2>

Pode-se concluir, após a verificação do gráfico de barras e a utilização do modelo, que o investimento em TV possui uma rentabilidade de ~85%. Já em Jornais, é menor do que 5%. É possível afirmar que, para esse caso, o investimento em TV trará mais retorno.
