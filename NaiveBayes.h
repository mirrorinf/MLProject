//
// Created by ChuXiaoMin on 2020/12/11.
//

#ifndef MLPROJECT_NAIVEBAYES_H
#define MLPROJECT_NAIVEBAYES_H

#include "WeightedClassifier.h"

// This is only for testing AdaBoost
class NaiveBayes : public WeightedClassifier {
public:
    std::vector<int> n_arrtribute_values;
    int N, K;
    std::vector<xt::xarray<double>> parameters;
    xt::xarray<double> y_probability;

    NaiveBayes(std::vector<int> n_attribute_possible_values, int n_classes);

    void train(xt::xarray<int> &x_train, std::vector<int> &y_train, xt::xarray<double> weight) override;
    std::vector<int> predict(xt::xarray<int> &x) override;
};


#endif //MLPROJECT_NAIVEBAYES_H
