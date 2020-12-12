//
// Created by ChuXiaoMin on 2020/12/10.
//

#include "Utility.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <map>

namespace MNISTUtility {
    int reverse_int(int i) {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

    xt::xarray<unsigned char> read_images(const std::string &path) {
        std::ifstream source(path, std::ios::binary);

        if (!source.is_open()) {
            throw std::runtime_error("Failed to open file " + path + ".");
        }
        int magic, n_row, n_column, n_images;
        source.read(reinterpret_cast<char *>(&magic), sizeof(int));
        magic = reverse_int(magic);

        if (magic != 2051) {
            throw std::runtime_error("Invalid file " + path + ".");
        }

        source.read(reinterpret_cast<char *>(&n_images), sizeof(int));
        n_images = reverse_int(n_images);

        source.read(reinterpret_cast<char *>(&n_row), sizeof(int));
        n_row = reverse_int(n_row);

        source.read(reinterpret_cast<char *>(&n_column), sizeof(int));
        n_column = reverse_int(n_column);

        xt::xarray<unsigned char> buffer = xt::zeros<unsigned char>({n_images, n_row, n_column});
        source.read(reinterpret_cast<char *>(buffer.data()), sizeof(unsigned char) * n_images * n_row * n_column);

        return buffer;
    }

    std::vector<int> read_labels(const std::string &path) {
        std::ifstream source(path, std::ios::binary);

        if (!source.is_open()) {
            throw std::runtime_error("Failed to open file " + path + ".");
        }
        int magic, n_labels;
        source.read(reinterpret_cast<char *>(&magic), sizeof(int));
        magic = reverse_int(magic);

        if (magic != 2049) {
            throw std::runtime_error("Invalid file " + path + ".");
        }

        source.read(reinterpret_cast<char *>(&n_labels), sizeof(int));
        n_labels = reverse_int(n_labels);

        std::vector<unsigned char> buffer(n_labels, 0);
        source.read(reinterpret_cast<char *>(buffer.data()), sizeof(unsigned char) * n_labels);

        std::vector<int> result(buffer.begin(), buffer.end());
        return result;
    }

    xt::xarray<int> binary_split(xt::xarray<unsigned char> &x) {
        const auto& shape = x.shape();
        xt::xarray<int> splitted = xt::zeros<int>(shape);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                splitted(i, j) = (x(i, j) > 127 ? 1 : 0);
            }
        }
        return splitted;
    }
}

namespace MushroomUnility {
    std::tuple<xt::xarray<int>, std::vector<int>, std::vector<int>> read_dataset(const std::string &path) {
        std::ifstream source(path);
        if (!source.is_open()) {
            throw std::runtime_error("Failed to open file " + path + ".");
        }

        auto raw = xt::load_csv<char>(source);
        const auto& s = raw.shape();
        if (s.size() != 2 || s[0] != 8124 || s[1] != 23) {
            throw std::runtime_error("Invalid mushroom data.");
        }

        std::vector<int> y_train(8124, 0);
        for (int i = 0; i < 8124; i++) {
            if (raw(i, 0) == 'e') {
                y_train[i] = 0;
            } else {
                y_train[i] = 1;
            }
        }

        xt::xarray<int> x_train = xt::zeros<int>({8124, 21});
        std::vector<int> n_value(21, 0);
        for (int i = 0; i < 21; i++) {
            int index;
            if (i < 9) {
                index = i + 1;
            } else {
                index = i + 2;
            }
            int n_v = 0;
            std::map<char, int> convert_to_value;
            for (int j = 0; j < 8124; j++) {
                if (!convert_to_value.contains(raw(j, index))) {
                    convert_to_value[raw(j, index)] = n_v;
                    n_v++;
                }
            }
            n_value[i] = n_v;
            for (int j = 0; j < 8124; j++) {
                x_train(j, i) = convert_to_value.at(raw(j, index));
            }
        }

        return std::make_tuple(x_train, y_train, n_value);
    }
}
