
# README - Treinamento de Modelo ADALINE para Classificação Binária

**Membros do grupo**: 

- Daniel Lucas Murta
- Davi Lorenzo Bambirra Braga
- Felipe Augusto Morais Silva
- Pedro Henrique Lopes Costa

Este projeto implementa a aprendizagem de um modelo **ADALINE** (Adaptive Linear Neuron) para classificação binária com base no modelo fornecido em https://github.com/TheAlgorithms/C/blob/e5dad3fa8def3726ec850ca66a7f51521f8ad393/machine_learning/adaline_learning.c. O modelo é treinado usando um arquivo de dados CSV, onde as características são usadas para prever a classe binária de um dataset sobre a qualidade de determinados vinhos (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?resource=download).

## Funcionalidades:

- **Leitura de arquivo CSV**: O código lê um arquivo CSV contendo características e valores de classificação.
- **Classificação binária**: A coluna "quality" (penúltima coluna) é convertida em uma classificação binária com base em um limiar (default é 6.0).
- **Treinamento do modelo ADALINE**: O modelo é treinado usando as características e os valores de classificação do conjunto de dados.
- **Predição manual**: O usuário pode fornecer uma entrada manual para obter a previsão do modelo.

## Pré-requisitos:

- **Compilador C++** (como o `g++` ou `clang++`)
- **Biblioteca padrão C++** (não é necessário instalar nada extra)

## Estrutura do Arquivo CSV:

O arquivo CSV deve ter a seguinte estrutura:
- **Colunas de características**: Os dados de entrada para o modelo.
- **Coluna "quality"**: A penúltima coluna, usada para a classificação binária.

Exemplo de um arquivo CSV:

```
feature1,feature2,feature3,...,quality
5.1,3.5,1.4,...,7.0
4.9,3.0,1.4,...,6.0
6.2,2.9,4.3,...,5.0
...
```

A coluna "quality" é convertida em uma classificação binária:
- Valores **maiores ou iguais a 6.0** são classificados como **1** (positivo).
- Valores **menores que 6.0** são classificados como **0** (negativo).

## Compilação do Código:

1. **Abra o terminal ou prompt de comando** no diretório onde o código está localizado.
2. **Compile o código** com o seguinte comando:

   ```bash
   g++ -o adaline adaline.cpp -std=c++11
   ```

   Isso irá gerar o arquivo executável `adaline`.

## Execução do Código:

Após compilar o código, você pode **executar** o programa com o seguinte comando:

```bash
./adaline <arquivo_dados.csv> <taxa_de_aprendizado>
```

Onde:
- `<arquivo_dados.csv>` é o arquivo CSV contendo os dados de entrada.
- `<taxa_de_aprendizado>` é o valor da taxa de aprendizado (eta) do modelo. Este valor deve estar entre 0 e 1. Exemplo: `0.01`.

### Exemplo de Execução:

```bash
./adaline dados.csv 0.01
```

Isso executará o programa com o arquivo `dados.csv` e a taxa de aprendizado de `0.01`.

## Testando com Entrada Manual:

Após o treinamento do modelo, o programa pedirá para o usuário fornecer uma entrada manual. Você deve digitar os valores das características para prever a classe binária para essa entrada.

Exemplo de entrada:

```
Insira os valores para previsão (3 valores separados por espaço):
5.6 3.2 1.4
```

O programa então irá retornar a predição do modelo para essa entrada.

## Estrutura do Código:

1. **Função `load_dataset`**: Lê o arquivo CSV, processa os dados e separa as características e as classes binárias.
2. **Função `main`**:
   - Recebe o arquivo CSV e a taxa de aprendizado como parâmetros.
   - Chama `load_dataset` para carregar os dados.
   - Treina o modelo ADALINE com as características e as classes.
   - Permite ao usuário testar o modelo com entradas manuais.
3. **Modelo ADALINE**:
   - Treinamento baseado nas entradas e valores binários.
   - Predição para novas entradas.

## Detalhes do Modelo ADALINE:

ADALINE é um modelo de rede neural de camada única que utiliza uma regra de aprendizado adaptativa para ajustar os pesos com base no erro da predição. Ele é usado principalmente para problemas de regressão, mas também pode ser adaptado para classificação binária, como neste caso.

## Problemas Conhecidos:

- O código assume que o arquivo CSV está no formato correto, com a coluna "quality" no penúltimo lugar. Se o arquivo não seguir esse formato, o código pode gerar erros de leitura.
- A função de entrada manual é baseada em um número fixo de características. Certifique-se de fornecer exatamente o número de valores que o modelo espera.

## Licença:

Este código é fornecido para fins educacionais. Sinta-se à vontade para modificá-lo e usá-lo conforme necessário.
