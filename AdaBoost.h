//
// Created by ChuXiaoMin on 2020/12/10.
//

#ifndef MLPROJECT_ADABOOST_H
#define MLPROJECT_ADABOOST_H

#include "WeightedClassifier.h"

class AdaBoost {
public:
    AdaBoost(int n_classes, FeatureList &x_train, std::vector<int> &y_train,
                       xt::xarray<double> &initial_weight,
                       std::vector<std::unique_ptr<WeightedClassifier>> &weak_classifiers,
                       FeatureList &x_validation, std::vector<int> &y_validation,
                       xt::xarray<double> &validation_weight);
    std::vector<int> predict(FeatureList &x);
private:
    int K, M;
    xt::xarray<double> weight;
    std::vector<double> alpha;
    std::vector<std::unique_ptr<WeightedClassifier>> &weak_classifiers;
};

#endif //MLPROJECT_ADABOOST_H
