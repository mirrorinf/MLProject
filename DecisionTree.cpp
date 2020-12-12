//
// Created by ChuXiaoMin on 2020/12/1.
//

#include "DecisionTree.h"
#include <iterator>
#include <algorithm>
#include <tuple>

namespace DecisionTreeUtility {
    typedef xt::xarray<int> Feature;
    typedef std::tuple<double, Feature, int> Sample;
    typedef std::vector<Sample>::iterator SampleIndex;

    int choose_split(const std::vector<int> &n_value, SampleIndex begin, SampleIndex end,
                     std::shared_ptr<DecisionTree::DecisionTreeNode> current_root);

    void split(int feature_index, const std::vector<int> &n_value, SampleIndex begin, SampleIndex end,
               std::shared_ptr<DecisionTree::DecisionTreeNode> current_root) {
        std::sort(begin, end, [feature_index](Sample x, Sample y) {
            auto vec_x = std::get<1>(x);
            auto vec_y = std::get<1>(y);
            return vec_x(feature_index) < vec_y(feature_index);
        });

        auto previous = begin;
        auto temp = std::get<1>(*begin);
        auto previous_value = temp(feature_index);
        for (auto i = begin + 1; i < end; i++) {
            temp = std::get<1>(*i);
            if (temp(feature_index) > previous_value) {
                auto ptr = std::shared_ptr<DecisionTree::DecisionTreeNode>(new DecisionTree::DecisionTreeNode);
            }
        }
    }
}

DecisionTree::DecisionTree(int n_classes, std::vector<int> n_v) : K(n_classes), n_value(n_v) {
    root = std::shared_ptr<DecisionTreeNode>(new DecisionTreeNode());
}

std::vector<int> DecisionTree::predict(xt::xarray<int> &x) {
    return WeightedClassifier::predict(x);
}

void DecisionTree::train(xt::xarray<int> &x_train, std::vector<int> &y_train, xt::xarray<double> weight) {

}
