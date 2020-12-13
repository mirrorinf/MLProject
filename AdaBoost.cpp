//
// Created by ChuXiaoMin on 2020/12/10.
//

#include "AdaBoost.h"
#include <cmath>
#include <iostream>

AdaBoost::AdaBoost(int n_classes, FeatureList &x_train, std::vector<int> &y_train,
                   xt::xarray<double> &initial_weight,
                   std::vector<std::unique_ptr<WeightedClassifier>> &weak_classifiers,
                   FeatureList &x_validation, std::vector<int> &y_validation,
                   xt::xarray<double> &validation_weight)
        : weak_classifiers(weak_classifiers) {
    K = n_classes;
    int n_classifier = weak_classifiers.size();
    int N = y_train.size();
    weight = initial_weight;

    for (int i = 0; i < weak_classifiers.size(); i++) {
        std::cout << "training weak No." << i << std::endl;
        weak_classifiers[i]->train(x_train, y_train, weight);
        std::cout << "predicting on weak No." << i << std::endl;
        auto y_predicted = weak_classifiers[i]->predict(x_train);

        xt::xarray<int> is_wrong = xt::zeros<int>({N});
        double error = 1e-8;
        for (int j = 0; j < N; j++) {
            if (y_predicted[j] != y_train[j]) {
                error += weight[j];
                is_wrong[j] = 1;
            }
        }

        double round_alpha = log((1 - error) / error) + log(K - 1);
        alpha.push_back(round_alpha);
        std::cout << "error of round " << i << ": " << error << ", alpha: " << round_alpha << std::endl;


        xt::xarray<double> adaptive = xt::exp(round_alpha * is_wrong);
        weight *= adaptive;
        double total = xt::sum(weight)();
        weight /= total;

        std::cout << "evaluating after round No." << i;
        M = i + 1;
        double eval_correct = 0;
        auto s = predict(x_validation);
        for (int i = 0; i < y_validation.size(); i++) {
            if (s[i] == y_validation[i]) {
                eval_correct += validation_weight[i];
            }
        }
        std::cout << " accuracy: " << eval_correct << std::endl;
    }
}

std::vector<int> AdaBoost::predict(FeatureList &x) {
    std::vector<std::vector<int>> indiviual_results;
    indiviual_results.reserve(M);
    for (int i = 0; i < M; i++) {
        indiviual_results.push_back(weak_classifiers[i]->predict(x));
    }
    int n_samples = indiviual_results[0].size();
    xt::xarray<double> count = xt::zeros<double>({n_samples, K});
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n_samples; j++) {
            count(j, indiviual_results[i][j]) += alpha[i];
        }
    }
    xt::xarray<int> temp = xt::argmax(count, 1);
    return std::vector<int>(temp.begin(), temp.end());
}