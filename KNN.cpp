//
// Created by ChuXiaoMin on 2020/12/10.
//

#include "KNN.h"
#include <xtensor-blas/xlinalg.hpp>
#include <assert.h>
#include <algorithm>

KNN::KNN(int n_classes, xt::xarray<double> stored_examples_x, std::vector<int> stored_examples_y) {
    // check shapes
    const auto& s = stored_examples_x.shape();
    assert(s.size() == 2);
    assert(s[0] == stored_examples_y.size());

    auto x_squared = stored_examples_x * stored_examples_x;
    xt::xarray<double> x_normalize_factor = xt::sqrt(xt::sum(x_squared, 1));
    x_normalize_factor.reshape({s[0], 1});
    xt::xarray<double> x_normalized = stored_examples_x / x_normalize_factor;

    this->n_classes = n_classes;
    this->stored_examples_x = xt::transpose(x_normalized, {1, 0});
    this->stored_examples_y = stored_examples_y;
}

std::vector<std::vector<int>> KNN::predict(xt::xarray<double> x, std::vector<int> ks) {
    // check shapes
    const auto& s1 = x.shape();
    const auto& s2 = stored_examples_x.shape();
    assert(s1.size() == 2);
    assert(s1[1] == s2[0]);

    int M = s1[0];
    xt::xarray<double> cross_distance = xt::linalg::dot(x, stored_examples_x);

    std::cout << "cross distance." << std::endl;

    int K = *std::max_element(ks.begin(), ks.end());

    std::vector<std::vector<int>> choices;
    for (int i = 0; i < K; i++) {
        std::cout << "nearest: " << i << std::endl;
        xt::xarray<int> max_indices = xt::argmax(cross_distance, 1);
        std::vector<int> chosen(M, 0);
        for (int j = 0; j < M; j++) {
            cross_distance(j, max_indices(j)) = -1e128;
            chosen[j] = stored_examples_y[max_indices(j)];
        }
        choices.push_back(chosen);
    }

    xt::xarray<int> count = xt::zeros<int>({M, n_classes});
    std::vector<std::vector<int>> result;
    for (int i = 0; i < K; i++) {
        std::cout << "accumulate: " << i << std::endl;
        for (int j = 0; j < M; j++) {
            count(j, choices[i][j])++;
        }
        if (std::find(ks.begin(), ks.end(), i + 1) != ks.end()) {
            xt::xarray<int> values = xt::argmax(count, 1);
            auto round_result = std::vector<int>(values.begin(), values.end());
            result.push_back(round_result);
        }
    }

    return result;
}
