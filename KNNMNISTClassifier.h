//
// Created by ChuXiaoMin on 2020/12/10.
//

#ifndef MLPROJECT_KNNMNISTCLASSIFIER_H
#define MLPROJECT_KNNMNISTCLASSIFIER_H

#include <xtensor.hpp>

// Use angle-cosine distance
class KNNMNISTClassifier {
public:
    int k, n_classes;
    xt::xarray<double> stored_examples_x;
    std::vector<int> stored_examples_y;

    KNNMNISTClassifier(int n_classes, xt::xarray<double> stored_examples_x, std::vector<int> stored_examples_y);

    // ks must be in strict increasing order
    std::vector<std::vector<int>> predict(xt::xarray<double> x, std::vector<int> ks);
};


#endif //MLPROJECT_KNNMNISTCLASSIFIER_H
