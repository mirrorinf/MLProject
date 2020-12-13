//
// Created by ChuXiaoMin on 2020/12/1.
//

#ifndef MLPROJECT_DECISIONTREE_H
#define MLPROJECT_DECISIONTREE_H

#include <vector>
#include <xtensor.hpp>
#include "WeightedClassifier.h"

class DecisionTree: public WeightedClassifier {
public:
    class DecisionTreeNode {
    public:
        int attribute_index, attribute_value;

        bool is_leaf;
        // only valid when is leaf
        int prediction;
        // only valid when is not leaf
        std::vector<std::shared_ptr<class DecisionTreeNode>> children;

        DecisionTreeNode() {
            attribute_index = -1;
            attribute_value = -2333;
            is_leaf = false;
            prediction = -1;
        }
    };

    int K;
    std::vector<int> n_value;
    std::shared_ptr<DecisionTreeNode> root;

    DecisionTree(int n_classes, std::vector<int> n_v);
    void train(xt::xarray<int> &x_train, std::vector<int> &y_train, xt::xarray<double> weight) override;
    std::vector<int> predict(xt::xarray<int> &x) override;
};


#endif //MLPROJECT_DECISIONTREE_H
