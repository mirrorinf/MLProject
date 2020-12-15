//
// Created by ChuXiaoMin on 2020/12/10.
//

#ifndef MLPROJECT_UTILITY_H
#define MLPROJECT_UTILITY_H

#include <xtensor.hpp>
#include <vector>
#include <string>
#include <tuple>

namespace MNISTUtility {
    xt::xarray<unsigned char> read_images(const std::string &path);
    std::vector<int> read_labels(const std::string &path);
    xt::xarray<int> binary_split(xt::xarray<unsigned char> &x);
    xt::xarray<int> pooled_binary_split(xt::xarray<unsigned char> &x);
}

namespace MushroomUnility {
    // x_train, y_train, n_value
    std::tuple<xt::xarray<int>, std::vector<int>, std::vector<int>> read_dataset(const std::string &path);
}

namespace SST2Utility {
    xt::xarray<int> split(xt::xarray<double> &x);
}

#endif //MLPROJECT_UTILITY_H
