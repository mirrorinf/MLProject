//
// Created by ChuXiaoMin on 2020/12/10.
//

#ifndef MLPROJECT_WEIGHTEDCLASSIFIER_H
#define MLPROJECT_WEIGHTEDCLASSIFIER_H

#include <vector>
#include <xtensor.hpp>
#include <exception>

typedef xt::xarray<int> FeatureList;

class WeightedClassifier {
public:
    virtual void train(FeatureList &x_train, std::vector<int> &y_train, xt::xarray<double> weight) {
        throw std::runtime_error("not implemented.");
    }
    virtual std::vector<int> predict(FeatureList &x) {
        throw std::runtime_error("not implemented.");
    }
};


#endif //MLPROJECT_WEIGHTEDCLASSIFIER_H
