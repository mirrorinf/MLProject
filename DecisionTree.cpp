//
// Created by ChuXiaoMin on 2020/12/1.
//

#include "DecisionTree.h"
#include <iterator>
#include <algorithm>
#include <tuple>
#include <utility>
#include <iostream>
#include <exception>
#include <cmath>
#include <numeric>
#include <random>

namespace DecisionTreeUtility {
    typedef xt::xarray<int> Feature;
    typedef std::tuple<double, Feature, int> Sample;
    typedef std::vector<Sample>::iterator SampleIndex;

    double xlogy(double x, double y) {
        if (x < 1e-10) {
            return 0;
        }
        return x * log(y);
    }

    int choose_split(int n_classes, const std::vector<int> &n_value, SampleIndex begin, SampleIndex end) {
        double max = -1e128, min = 1e128;
        int maxi = 0;
        for (int i = 0; i < n_value.size(); i++) {
            xt::xarray<double> conditional_frequencies = xt::ones<double>({n_value[i], n_classes}) * 1e-8;
            for (auto j = begin; j < end; j++) {
                auto temp = std::get<1>(*j);
                int x = temp(i);
                int y = std::get<2>(*j);
                conditional_frequencies(x, y) += std::get<0>(*j);
            }
            xt::xarray<double> normalize = xt::sum(conditional_frequencies, {1});
            normalize.reshape({n_value[i], 1});
            xt::xarray<double> s1 = conditional_frequencies * xt::log(conditional_frequencies);
            xt::xarray<double> logp = xt::log(normalize);
            xt::xarray<double> s2 = conditional_frequencies * logp;
            double information_gain = xt::sum(s1)() - xt::sum(s2)();

            if (information_gain > max) {
                maxi = i;
                max = information_gain;
            }
            if (information_gain < min) {
                min = information_gain;
            }
        }
        // no feature is significantly better
        // stop splitting
        if (max - min < 1e-6) {
            return -1;
        }
        return maxi;
    }

    int argmax(std::vector<double> &x) {
        double max = x[0];
        int maxi = 0;
        for (int i = 1; i < x.size(); i++) {
            if (x[i] > max) {
                max = x[i];
                maxi = i;
            }
        }
        return maxi;
    }

    int number_of_non_zero_element(std::vector<double> &x) {
        int count = 0;
        for (auto i = x.begin(); i < x.end(); i++) {
            if (*i != 0) {
                count++;
            }
        }
        return count;
    }

    // inconsistency in train data leads to infinite recursion
    // elimination of inconsistency in preprocessing is necessary
    void split(int n_classes, int feature_index, const std::vector<int> &n_value, SampleIndex begin, SampleIndex end,
               std::shared_ptr<DecisionTree::DecisionTreeNode> current_root) {
        current_root->attribute_index = feature_index;
        auto count = std::vector<double>(n_classes, 0);
        for (auto i = begin; i < end; i++) {
            auto this_class = std::get<2>(*i);
            count[this_class] += std::get<0>(*i);
        }
        current_root->prediction = argmax(count);

        if (feature_index < 0) {
            current_root->is_leaf = true;
            /*
            std::cout << "Node: no feature selection, " << "majority vote: "
                      << current_root->prediction << ", contain samples: "
                      << end - begin << ", total weight: "
                      << std::accumulate(count.begin(), count.end(), decltype(count)::value_type(0))
                      << " LEAF" << std::endl;
            */
            return;
        }
        /*
        std::cout << "Node: using feature: " << feature_index << ", majority vote: "
                  << current_root->prediction << ", contain samples: "
                  << end - begin << ", total weight: "
                  << std::accumulate(count.begin(), count.end(), decltype(count)::value_type(0));
        */
        if (number_of_non_zero_element(count) <= 1) {
            current_root->is_leaf = true;
            // std::cout << " PURE LEAF" << std::endl;
            return;
        }
        // std::cout << " INTERNAL" << std::endl;

        std::sort(begin, end, [feature_index](Sample x, Sample y) {
            auto vec_x = std::get<1>(x);
            auto vec_y = std::get<1>(y);
            return vec_x(feature_index) < vec_y(feature_index);
        });

        auto previous = begin;
        auto spliting = false;
        auto temp = std::get<1>(*begin);
        auto previous_value = temp(feature_index);
        current_root->children = std::vector<std::shared_ptr<DecisionTree::DecisionTreeNode>>(n_value[feature_index],
                                                                                                nullptr);
        for (auto i = begin + 1; i < end; i++) {
            temp = std::get<1>(*i);
            if (temp(feature_index) > previous_value) {
                spliting = true;
                auto ptr = std::shared_ptr<DecisionTree::DecisionTreeNode>(new DecisionTree::DecisionTreeNode);
                ptr->is_leaf = false;
                current_root->children[previous_value] = ptr;
                int new_index = choose_split(n_classes, n_value, previous, i);
                split(n_classes, new_index, n_value, previous, i, ptr);
                previous = i;
                previous_value = temp(feature_index);
            }
        }
        if (!spliting) {
            std::cout << "INCONSISTENCY. node forced to be leaf." << std::endl;
            current_root->is_leaf = true;
            return;
        }

        auto ptr = std::shared_ptr<DecisionTree::DecisionTreeNode>(new DecisionTree::DecisionTreeNode);
        ptr->is_leaf = false;
        current_root->children[previous_value] = ptr;
        int new_index = choose_split(n_classes, n_value, previous, end);
        split(n_classes, new_index, n_value, previous, end, ptr);
    }

    int single_predict(Feature &x, std::shared_ptr<DecisionTree::DecisionTreeNode> root) {
        std::shared_ptr<DecisionTree::DecisionTreeNode> current = root;
        while (!current->is_leaf && current->children[x(current->attribute_index)] != nullptr) {
            current = current->children[x(current->attribute_index)];
        }
        return current->prediction;
    }

    // returning weighted error
    double post_pruning(std::shared_ptr<DecisionTree::DecisionTreeNode> current, SampleIndex begin, SampleIndex end) {
        double error = 0;

        for (auto i = begin; i < end; i++) {
            if (std::get<2>(*i) != current->prediction) {
                error += std::get<0>(*i);
            }
        }

        if (current->is_leaf) {
            return error;
        }

        int feature_index = current->attribute_index;
        auto left = begin;
        double no_prune_error = 0;
        for (auto i = 0; i < current->children.size(); i++) {
            auto new_left = std::partition(left, end, [feature_index, i](Sample x) {
               auto vec = std::get<1>(x);
               return vec(feature_index) <= i;
            });
            if (current->children[i] == nullptr) {
                for (auto j = left; j < new_left; j++) {
                    if (std::get<2>(*j) != current->prediction) {
                        no_prune_error += std::get<0>(*j);
                    }
                }
            } else {
                no_prune_error += post_pruning(current->children[i], left, new_left);
            }
            left = new_left;
        }

        if (error < no_prune_error) {
            // std::cout << "INTERNAL error before pruning: " << no_prune_error << ", error after pruning: " << error;
            // std::cout << ", pruning Node: feature_index: " << current->attribute_index << std::endl;
            current->is_leaf = true;
            return error;
        }
        return no_prune_error;
    }
}

DecisionTree::DecisionTree(int n_classes, std::vector<int> n_v) : K(n_classes), n_value(std::move(n_v)) {
    root = std::shared_ptr<DecisionTreeNode>(new DecisionTreeNode());
}

std::vector<int> DecisionTree::predict(xt::xarray<int> &x) {
    const auto& shape = x.shape();
    std::vector<int> result(shape[0], 0);
    for (int i = 0; i < shape[0]; i++) {
        xt::xarray<int> row = xt::view(x, i, xt::all());
        result[i] = DecisionTreeUtility::single_predict(row, root);
    }
    return result;
}

void DecisionTree::train(xt::xarray<int> &x_train, std::vector<int> &y_train, xt::xarray<double> weight) {
    std::vector<DecisionTreeUtility::Sample> table;
    table.reserve(y_train.size());
    std::cout << "Building tree on " << y_train.size() << " samples." << std::endl;
    for (int i = 0; i < y_train.size(); i++) {
        auto data_row = std::make_tuple(weight[i], xt::view(x_train, i, xt::all()), y_train[i]);
        table.push_back(data_row);
    }

    auto rng = std::default_random_engine {};
    std::shuffle(table.begin(), table.end(), rng);

    int train_size = (y_train.size() / 5) * 4;
    int validation_size = y_train.size() - train_size;
    std::cout << "Train: " << train_size << " Validation: " << validation_size << std::endl;
    int index = DecisionTreeUtility::choose_split(K, n_value, table.begin(), table.begin() + train_size);
    DecisionTreeUtility::split(K, index, n_value, table.begin(), table.begin() + train_size, root);
    std::cout << "Post pruning using validation data" << std::endl;
    DecisionTreeUtility::post_pruning(root, table.begin() + train_size, table.end());
}
