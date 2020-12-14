//
// Created by ChuXiaoMin on 2020/12/10.
//

#ifndef MLPROJECT_KNN_H
#define MLPROJECT_KNN_H

#include <xtensor.hpp>

// Use angle-cosine distance
class KNN {
public:
    int k, n_classes;
    xt::xarray<double> stored_examples_x;
    std::vector<int> stored_examples_y;

    KNN(int n_classes, xt::xarray<double> stored_examples_x, std::vector<int> stored_examples_y);

    // ks must be in strict increasing order
    std::vector<std::vector<int>> predict(xt::xarray<double> x, std::vector<int> ks);
};


#endif //MLPROJECT_KNN_H
