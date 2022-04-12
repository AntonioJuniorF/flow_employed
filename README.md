# Descrição do problema e objetivo.
 
* Objetivo principal deste projeto será a criação de um modelo de machine learning para prever se uma pessoa que foi contratada por uma determinada empresa vai ficar mais de um ano trabalhando nela ou não.
 
* Os dados foram disponibilizados pela comunidade Flai. Onde eles são compostos pelas seguintes variáveis:
 
 * **func_sexo:** Sexo da pessoa.
 * **func_idade:** Idade do usuário.
 * **func_racacor:** Raça da pessoa segundo a classificação do IBGE.
 * **func_escolaridade:** Escolaridade das pessoas.
 * **func_uf:** Estado da pessoa
 * **func_deficiencia:** Se a pessoa é deficiente ou não.
 * **empresa_porte:** Número de funcionários da empresa.
 * **empresa_setor:** Setor que a empresa trabalha.
 * **contrato_horastrabalho:** Quantidade de horas por semana de trabalho.
 * **contrato_salario:** Valor do salário da pessoa na empresa.
 * **turnover_apos_1_ano:** Se a pessoa trabalhou lá mais de um ano ou não.
 
# Metodologia
 
Este projeto será baseado no processo padrão Cross-industry para mineração de dados (CRISP-DM). Uma ideia padrão sobre projeto de ciência de dados pode ser linear: preparação de dados, modelagem, avaliação e implantação. No entanto, quando usamos a metodologia CRISP-DM, um projeto de ciência de dados se torna uma forma circular. Mesmo quando termina em Deployment, o projeto pode ser reiniciado novamente por Business Understanding. Como isso pode ajudar?
 
 
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png" alt="Kitten" title="A cute kitten" width="430" height="430" />
</p>
 
Pode ajudar a evitar que o cientista de dados pare em uma etapa específica e perca tempo com ela. Quando todo o projeto estiver concluído, o cientista de dados pode retornar à etapa inicial e fazer todas as etapas novamente. Portanto, o objetivo principal é seguir círculos conforme a necessidade. 
 
# Pipeline
 
* Tratamento dos dados.
* Análise dos dados.
* Preparação dos dados.
* Feature Engineering.
* Modelagem dos modelos de Machine Learning.
* Ajuste dos hiperparâmetros.
* Avaliação do melhor modelo. 
* Equilíbrio das Classes do Target
* Reavaliando o melhor modelo.
* Deploy do modelo.
 
# Aquisição dos dados.
 
* Os dados foram disponibilizados pela Flai para que os seus possam participar da sua competição de machine learning.
 
# Descrição de cada módulo do projeto

### Primeiro módulo
Essa parte do projeto está encarregada de tratar os dados, avaliá-los e criar o modelo de machine learning. Esse estudo foi dividido nas seguintes partes:
* Parte 1: Conhecer as características de cada variável e realizar o tratamento dos dados.
* Parte 2: Seleção de variáveis para treinar o modelo e criar outras novas. E como a variação do limiar do modelo afeta no seu desempenho
* Parte 3: Modelagem do modelo de machine learning, interpretação dos seus resultados.
* Parte 4: Balanceamento das classes da variável de Target e como o modelo está performado depois disso.
 
**Todas essas partes se encontram na pasta chamada Notebooks**

### Segundo módulo
 Essa parte do projeto é o deploy do modelo. Onde ela foi desenvolvida através das seguintes algoritmos:
 
 * Model: Encarregado de salvar o modelo em disco e alguns tratamentos feitos nos dados.
 * Model_api: API desenvolvido com o Flask. 
 * Churn: Classe criada para processar os dados de entrada do API. 
 * Teste API: Notebook criado para testar o API. 

**O Model, Model_api se encontram na pasta API, Churn esta na pasta churn e o Teste API se encontra na pasta Notebooks** 


 

 
 
 

