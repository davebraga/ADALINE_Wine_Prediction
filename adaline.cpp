#include <omp.h>  // Adicionada a inclusão do cabeçalho OpenMP
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

/** TEMPOS
 * Sequencial: 0m9.838s
 * Paralelo (1T): 0m5.113s
 * Paralelo (2T): 0m3.872s
 * Paralelo (4T): 0m2.528s
 * Paralelo (8T): 0m1.431s
/**/

/**
 * @addtogroup machine_learning Machine learning algorithms
 * @{
 * @addtogroup adaline Adaline learning algorithm
 * @{
 */

/** Maximum number of iterations to learn */
#define MAX_ADALINE_ITER 500  // INT_MAX

/** structure to hold adaline model parameters */
struct adaline
{
    double eta;      /**< learning rate of the algorithm */
    double *weights; /**< weights of the neural network */
    int num_weights; /**< number of weights of the neural network */
};

/** convergence accuracy \f$=1\times10^{-5}\f$ */
#define ADALINE_ACCURACY 1e-5

/**
 * Default constructorada
 * \param[in] num_features number of features present
 * \param[in] eta learning rate (optional, default=0.1)
 * \returns new adaline model
 */
struct adaline new_adaline(const int num_features, const double eta)
{
    if (eta <= 0.f || eta >= 1.f)
    {
        fprintf(stderr, "learning rate should be > 0 and < 1\n");
        exit(EXIT_FAILURE);
    }

    // additional weight is for the constant bias term
    int num_weights = num_features + 1;
    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_weights;
    ada.weights = (double *)malloc(num_weights * sizeof(double));
    if (!ada.weights)
    {
        perror("Unable to allocate error for weights!");
        return ada;
    }

    // initialize with random weights in the range [-50, 49]
    for (int i = 0; i < num_weights; i++) ada.weights[i] = 1.f;
    // ada.weights[i] = (double)(rand() % 100) - 50);

    return ada;
}

/** delete dynamically allocated memory
 * \param[in] ada model from which the memory is to be freed.
 */
void delete_adaline(struct adaline *ada)
{
    if (ada == NULL)
        return;

    free(ada->weights);
};

/** [Heaviside activation
 * function](https://en.wikipedia.org/wiki/Heaviside_step_function) <img
 * src="https://upload.wikimedia.org/wikipedia/commons/d/d9/Dirac_distribution_CDF.svg"
 * width="200px"/>
 * @param x activation function input
 * @returns \f$f(x)= \begin{cases}1 & \forall\; x > 0\\ -1 & \forall\; x \le0
 * \end{cases}\f$
 */
int adaline_activation(double x) { return x > 0 ? 1 : -1; }

/**
 * Operator to print the weights of the model
 * @param ada model for which the values to print
 * @returns pointer to a NULL terminated string of formatted weights
 */
char *adaline_get_weights_str(const struct adaline *ada)
{
    static char out[100];  // static so the value is persistent

    sprintf(out, "<");
    for (int i = 0; i < ada->num_weights; i++)
    {
        sprintf(out, "%s%.4g", out, ada->weights[i]);
        if (i < ada->num_weights - 1)
            sprintf(out, "%s, ", out);
    }
    sprintf(out, "%s>", out);
    return out;
}

/**
 * predict the output of the model for given set of features
 *
 * \param[in] ada adaline model to predict
 * \param[in] x input vector
 * \param[out] out optional argument to return neuron output before applying
 * activation function (`NULL` to ignore)
 * \returns model prediction output
 */
int adaline_predict(struct adaline *ada, const double *x, double *out)
{
    double y = ada->weights[ada->num_weights - 1];  // assign bias value

    for (int i = 0; i < ada->num_weights - 1; i++) y += x[i] * ada->weights[i];

    if (out)  // if out variable is not NULL
        *out = y;

    // quantizer: apply ADALINE threshold function
    return adaline_activation(y);
}

/**
 * Update the weights of the model using supervised learning for one feature
 * vector
 *
 * \param[in] ada adaline model to fit
 * \param[in] x feature vector
 * \param[in] y known output  value
 * \returns correction factor
 */
double adaline_fit_sample(struct adaline *ada, const double *x, const int y)
{
    /* output of the model with current weights */
    int p = adaline_predict(ada, x, NULL);
    int prediction_error = y - p;  // error in estimation
    double correction_factor = ada->eta * prediction_error;

    /* update each weight, the last weight is the bias term */
    for (int i = 0; i < ada->num_weights - 1; i++)
    {
        ada->weights[i] += correction_factor * x[i];
    }
    ada->weights[ada->num_weights - 1] += correction_factor;  // update bias

    return correction_factor;
}

/**
 * Update the weights of the model using supervised learning for an array of
 * vectors.
 *
 * \param[in] ada adaline model to train
 * \param[in] X array of feature vector
 * \param[in] y known output value for each feature vector
 * \param[in] N number of training samples
 */

//A PARALELIZAÇÃO SERÁ FEITA NA FUNÇAO ABAIXO
// Alteração principal: Uso do OpenMP para paralelizar o loop que ajusta pesos em `adaline_fit`.
void adaline_fit(struct adaline *ada, double **X, const int *y, const int N) {
    double avg_pred_error = 1.f;
    int iter;

    for (iter = 0; (iter < MAX_ADALINE_ITER) && (avg_pred_error > ADALINE_ACCURACY); iter++) {
        avg_pred_error = 0.f;

        // Adicionado pragma para paralelizar o loop de amostras
        #pragma omp parallel for reduction(+:avg_pred_error) schedule(static)
        for (int i = 0; i < N; i++) {
            double err = adaline_fit_sample(ada, X[i], y[i]);
            avg_pred_error += fabs(err);
        }

        avg_pred_error /= N;

        // Print updates a cada 100 iterações (não paralelizado por simplicidade)
        if (iter % 100 == 0) {
            printf("\tIter %3d: Pesos: %s\tTaxa de erro: %.4f\n", iter, 
                   adaline_get_weights_str(ada), avg_pred_error);
        }
    }

    if (iter < MAX_ADALINE_ITER) {
        printf("Converged after %d iterations.\n", iter);
    } else {
        printf("Did not converge after %d iterations.\n", iter);
    }
}

// Leitura do dataset com simplificação binária para a coluna "quality"
void load_dataset(const char* filename, std::vector<std::vector<double>>& X, std::vector<int>& Y, double threshold = 6.0) {
    std::ifstream file(filename);
    std::string line;

    // Verifica se o arquivo foi aberto corretamente
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Lê o arquivo CSV linha por linha
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Lê os valores separados por vírgula
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value)); // Converte o valor de string para double
            } catch (const std::invalid_argument& e) {
                std::cerr << "Erro ao converter valor para número: " << value << std::endl;
            }
        }

        // Verifica se a linha tem pelo menos duas colunas (features + quality)
        if (row.size() < 2) {
            std::cerr << "Erro: linha inválida no arquivo!" << std::endl;
            continue;
        }

         // O penúltimo valor é o 'quality' (label)
        double quality = row[row.size() - 2]; // Coluna do quality
        row.pop_back(); // Remove o último valor (não necessário)
        row.pop_back(); // Remove o penúltimo valor (quality)

        // Adiciona as features e o label binário
        X.push_back(row);
        Y.push_back(quality >= threshold ? 1 : 0); // Limiar para classificação binária
    }

    file.close();
}

/** Main function - Receberá um arquivo .txt de entrada de dados */
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <arquivo_de_dados> <taxa_de_aprendizado>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* dataset_file = argv[1];
    double eta = atof(argv[2]);

    if (eta <= 0.0 || eta >= 1.0) {
        fprintf(stderr, "A taxa de aprendizado (eta) deve estar entre 0 e 1.\n");
        return EXIT_FAILURE;
    }

    std::vector<std::vector<double>> X;
    std::vector<int> Y;

    // Carregar a base de dados com simplificação para binário
    double threshold = 6.0; // Limiar para classificação binária
    load_dataset(dataset_file, X, Y, threshold);

    int num_samples = X.size();
    int num_features = num_samples > 0 ? X[0].size() : 0;

    if (num_features == 0) {
        fprintf(stderr, "O arquivo de dados esta vazio ou nao possui características validas.\n");
        return EXIT_FAILURE;
    }

    // Alocar matriz para treinamento
    double** X_matrix = (double**)malloc(num_samples * sizeof(double*));
    for (int i = 0; i < num_samples; i++) {
        X_matrix[i] = (double*)malloc(num_features * sizeof(double));
        for (int j = 0; j < num_features; j++) {
            X_matrix[i][j] = X[i][j];
        }
    }

    // Treinar o modelo
    struct adaline ada = new_adaline(num_features, eta);
    printf("Modelo antes do treinamento: %s\n", adaline_get_weights_str(&ada));
    adaline_fit(&ada, X_matrix, Y.data(), num_samples);
    printf("Modelo apos o treinamento: %s\n", adaline_get_weights_str(&ada));

    // Testar uma entrada manual
    std::vector<double> test_input(num_features);
    printf("Insira os valores para previsao (%d valores separados por espaco):\n", num_features);
    for (int i = 0; i < num_features; i++) {
        std::cin >> test_input[i];
    }

    int prediction = adaline_predict(&ada, test_input.data(), nullptr);
    printf("Predicao para a entrada fornecida: %d\n", prediction);

    // Liberar memória
    for (int i = 0; i < num_samples; i++) free(X_matrix[i]);
    free(X_matrix);
    delete_adaline(&ada);

    return EXIT_SUCCESS;
}