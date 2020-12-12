//
// Created by ChuXiaoMin on 2020/12/11.
//

#include "NaiveBayes.h"

NaiveBayes::NaiveBayes(std::vector<int> n_attribute_possible_values, int n_classes) {
    n_arrtribute_values = n_attribute_possible_values;
    K = n_classes;
    N = n_attribute_possible_values.size();
    y_probability = xt::zeros<double>({N});
}

void NaiveBayes::train(xt::xarray<int> &x_train, std::vector<int> &y_train, xt::xarray<double> weight) {
    int M = y_train.size();
    for (int i = 0; i < M; i++) {
        y_probability[y_train[i]] += weight[i];
    }

    for (int i = 0; i < N; i++) {
        xt::xarray<double> conditional_probability = xt::zeros<double>({n_arrtribute_values[i], K});
        for (int j = 0; j < M; j++) {
            conditional_probability(x_train(j, i), y_train[j]) += weight[j];
        }
        xt::xarray<double> normalize = xt::sum(conditional_probability, {0});
        normalize.reshape({1, K});
        conditional_probability /= normalize;
        parameters.push_back(conditional_probability);
    }
}

std::vector<int> NaiveBayes::predict(xt::xarray<int> &x) {
    const auto& size = x.shape();
    int M = size[0];

    xt::xarray<double> probs = xt::ones<double>({M, K});
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            xt::view(probs, j, xt::all()) *= xt::view(parameters[i], x(j, i), xt::all());
        }
    }

    xt::xarray<int> temp = xt::argmax(probs, {1});
    return std::vector(temp.begin(), temp.end());
}
