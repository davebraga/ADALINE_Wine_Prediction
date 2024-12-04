#include <omp.h>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits.h>

/** Constantes */
#define MAX_ADALINE_ITER 500
#define ADALINE_ACCURACY 1e-5

/** Estrutura do modelo Adaline */
struct adaline {
    double eta;      // Taxa de aprendizado
    double *weights; // Pesos do modelo
    int num_weights; // Número de pesos (incluindo bias)
};

/** Função para inicializar o modelo Adaline */
struct adaline new_adaline(const int num_features, const double eta) {
    if (eta <= 0.f || eta >= 1.f) {
        fprintf(stderr, "Taxa de aprendizado deve estar entre 0 e 1.\n");
        exit(EXIT_FAILURE);
    }

    struct adaline ada;
    ada.eta = eta;
    ada.num_weights = num_features + 1;
    ada.weights = (double *)malloc(ada.num_weights * sizeof(double));
    if (!ada.weights) {
        perror("Erro ao alocar memória para os pesos!");
        exit(EXIT_FAILURE);
    }

    // Inicializar pesos com valores aleatórios
    for (int i = 0; i < ada.num_weights; i++) {
        ada.weights[i] = (double)(rand() % 100) / 100.0 - 0.5; // [-0.5, 0.5]
    }

    return ada;
}

/** Liberação de memória do modelo Adaline */
void delete_adaline(struct adaline *ada) {
    if (ada != NULL) {
        free(ada->weights);
    }
}

/** Função de ativação Heaviside */
int adaline_activation(double x) {
    return x > 0 ? 1 : 0;
}

/** Retorna os pesos do modelo como string */
char *adaline_get_weights_str(const struct adaline *ada) {
    static char out[1000];
    sprintf(out, "<");
    for (int i = 0; i < ada->num_weights; i++) {
        sprintf(out + strlen(out), "%.4f", ada->weights[i]);
        if (i < ada->num_weights - 1) {
            sprintf(out + strlen(out), ", ");
        }
    }
    sprintf(out + strlen(out), ">");
    return out;
}

/** Prediz o valor com base no modelo Adaline */
int adaline_predict(struct adaline *ada, const double *x, double *out) {
    double y = ada->weights[ada->num_weights - 1]; // Bias
    for (int i = 0; i < ada->num_weights - 1; i++) {
        y += x[i] * ada->weights[i];
    }

    if (out) *out = y;
    return adaline_activation(y);
}

/** Ajusta os pesos para uma amostra */
double adaline_fit_sample(struct adaline *ada, const double *x, const int y) {
    double prediction_error = y - adaline_predict(ada, x, nullptr);
    double correction_factor = ada->eta * prediction_error;

    // Atualizar os pesos
    for (int i = 0; i < ada->num_weights - 1; i++) {
        ada->weights[i] += correction_factor * x[i];
    }
    ada->weights[ada->num_weights - 1] += correction_factor; // Atualizar bias

    return fabs(correction_factor);
}

/** Ajusta os pesos para múltiplas amostras (treinamento paralelo) */
void adaline_fit(struct adaline *ada, double **X, const int *y, const int N) {
    double avg_pred_error = 1.0;
    int iter;

    for (iter = 0; iter < MAX_ADALINE_ITER && avg_pred_error > ADALINE_ACCURACY; iter++) {
        avg_pred_error = 0.0;

        #pragma omp parallel for reduction(+:avg_pred_error)
        for (int i = 0; i < N; i++) {
            avg_pred_error += adaline_fit_sample(ada, X[i], y[i]);
        }

        avg_pred_error /= N;

        if (iter % 100 == 0) {
            printf("Iteração %d: Pesos: %s Erro médio: %.4f\n",
                   iter, adaline_get_weights_str(ada), avg_pred_error);
        }
    }

    printf("Treinamento %s após %d iterações.\n",
           (iter < MAX_ADALINE_ITER) ? "convergiu" : "não convergiu", iter);
}

/** Carrega o dataset */
void load_dataset(const char* filename, std::vector<std::vector<double>>& X, std::vector<int>& Y, double threshold = 6.0) {
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo!" << std::endl;
        exit(EXIT_FAILURE);
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        double quality = row.back();
        row.pop_back();

        X.push_back(row);
        Y.push_back(quality >= threshold ? 1 : 0);
    }

    file.close();
}

/** Função principal */
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
    load_dataset(dataset_file, X, Y);

    int num_samples = X.size();
    int num_features = X[0].size();

    double** X_matrix = (double**)malloc(num_samples * sizeof(double*));
    for (int i = 0; i < num_samples; i++) {
        X_matrix[i] = (double*)malloc(num_features * sizeof(double));
        for (int j = 0; j < num_features; j++) {
            X_matrix[i][j] = X[i][j];
        }
    }

    struct adaline ada = new_adaline(num_features, eta);
    printf("Modelo antes do treinamento: %s\n", adaline_get_weights_str(&ada));
    adaline_fit(&ada, X_matrix, Y.data(), num_samples);
    printf("Modelo após o treinamento: %s\n", adaline_get_weights_str(&ada));

    // Liberar memória
    for (int i = 0; i < num_samples; i++) free(X_matrix[i]);
    free(X_matrix);
    delete_adaline(&ada);

    return EXIT_SUCCESS;
}
