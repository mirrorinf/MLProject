#include "Utility.h"
#include "KNN.h"
#include <iostream>
#include "NaiveBayes.h"
#include "AdaBoost.h"
#include <algorithm>
#include "DecisionTree.h"
#include <xtensor/xnpy.hpp>

void mnist_knn_main() {
    auto data_x = MNISTUtility::read_images("/Users/chuxiaomin/Develop/ai/ml/train-images-idx3-ubyte");
    auto data_y = MNISTUtility::read_labels("/Users/chuxiaomin/Develop/ai/ml/train-labels-idx1-ubyte");

    xt::xarray<double> x = xt::cast<double>(data_x);
    x /= 255;
    x.reshape({60000, 28 * 28});

    xt::xarray<double> x_train = xt::view(x, xt::range(_, 50000));
    xt::xarray<double> x_validation = xt::view(x, xt::range(50000, _));
    auto y_train = std::vector<int>(data_y.begin(), data_y.begin() + 50000);
    auto y_validation = std::vector<int>(data_y.begin() + 50000, data_y.end());

    auto knn = KNN(10, x_train, y_train);
    std::vector<int> ks(20);
    std::iota(ks.begin(), ks.end(), 1);
    auto predicted = knn.predict(x_validation, ks);

    for (int i : ks) {
        int correct = 0;
        for (int j = 0; j < 10000; j++) {
            if (predicted[i - 1][j] == y_validation[j]) {
                correct++;
            }
        }
        std::cout << "k: " << i << "  correct:" << correct << std::endl;
    }
}

void mushroom_test_main() {
    auto s = MushroomUnility::read_dataset("/Users/chuxiaomin/Develop/ai/ml/agaricus-lepiota.data");
    auto x_whole = std::get<0>(s);
    auto y_whole = std::get<1>(s);
    auto n_value = std::get<2>(s);

    xt::xarray<int> x_train = xt::view(x_whole, xt::range(_, 6000), xt::all());
    xt::xarray<int> x_validation = xt::view(x_whole, xt::range(6000, _), xt::all());
    auto y_train = std::vector<int>(y_whole.begin(), y_whole.begin() + 6000);
    auto y_validation = std::vector<int>(y_whole.begin() + 6000, y_whole.end());

    xt::xarray<double> weight = xt::ones<double>({6000}) * (1.0 / 6000);

    {
        auto classifier = NaiveBayes(n_value, 2);
        classifier.train(x_train, y_train, weight);
        auto predicted = classifier.predict(x_validation);
        int correct = 0;
        for (int i = 0; i < predicted.size(); i++) {
            if (predicted[i] == y_validation[i]) {
                correct++;
            }
        }
        printf("single: %lf\n", static_cast<double>(correct) / predicted.size());
    }
    {
        xt::xarray<double> validation_weight = xt::ones<double>({y_validation.size()}) * (1.0 / y_validation.size());
        std::vector<std::unique_ptr<WeightedClassifier>> weaks;
        weaks.reserve(5);
        for (int i = 0; i < 5; i++) {
            weaks.push_back(std::unique_ptr<WeightedClassifier>(new NaiveBayes(n_value, 2)));
        }
        auto boosted = AdaBoost(2, x_train, y_train, weight, weaks, x_validation, y_validation, validation_weight);
    }
}
/*
void mnist_naive_bayes_main() {
    auto data_x = MNISTUtility::read_images("/Users/chuxiaomin/Develop/ai/ml/train-images-idx3-ubyte");
    auto data_y = MNISTUtility::read_labels("/Users/chuxiaomin/Develop/ai/ml/train-labels-idx1-ubyte");
    data_x.reshape({60000, 28 * 28});

    xt::xarray<int> x_whole = MNISTUtility::binary_split(data_x);
    xt::xarray<int> x_train = xt::view(x_whole, xt::range(_, 50000), xt::all());
    xt::xarray<int> x_validation = xt::view(x_whole, xt::range(50000, _), xt::all());
    auto y_train = std::vector<int>(data_y.begin(), data_y.begin() + 50000);
    auto y_validation = std::vector<int>(data_y.begin() + 50000, data_y.end());

    xt::xarray<double> weight = xt::ones<double>({50000}) * (1.0 / 50000);

    auto n_value = std::vector<int>(28 * 28, 2);

    {
        printf("start training single.\n");
        auto classifier = NaiveBayes(n_value, 10);
        classifier.train(x_train, y_train, weight);
        auto predicted = classifier.predict(x_validation);
        int correct = 0;
        for (int i = 0; i < predicted.size(); i++) {
            if (predicted[i] == y_validation[i]) {
                correct++;
            }
        }
        printf("single: %lf\n", static_cast<double>(correct) / predicted.size());
    }
    {
        printf("start training boosted.\n");
        std::vector<std::unique_ptr<WeightedClassifier>> weaks;
        weaks.reserve(3);
        for (int i = 0; i < 3; i++) {
            weaks.push_back(std::unique_ptr<WeightedClassifier>(new NaiveBayes(n_value, 10)));
        }
        auto boosted = AdaBoost(10, x_train, y_train, weight, weaks);
        auto predicted = boosted.predict(x_validation);
        int correct = 0;
        for (int i = 0; i < predicted.size(); i++) {
            if (predicted[i] == y_validation[i]) {
                correct++;
            }
        }
        printf("boosted: %lf\n", static_cast<double>(correct) / predicted.size());
    }
}
*/
void mushroom_tree_main() {
    auto s = MushroomUnility::read_dataset("/Users/chuxiaomin/Develop/ai/ml/agaricus-lepiota.data");
    auto x_whole = std::get<0>(s);
    auto y_whole = std::get<1>(s);
    auto n_value = std::get<2>(s);

    xt::xarray<int> x_train = xt::view(x_whole, xt::range(_, 6000), xt::all());
    xt::xarray<int> x_validation = xt::view(x_whole, xt::range(6000, _), xt::all());
    auto y_train = std::vector<int>(y_whole.begin(), y_whole.begin() + 6000);
    auto y_validation = std::vector<int>(y_whole.begin() + 6000, y_whole.end());
    xt::xarray<double> weight = xt::ones<double>({6000}) * (1.0 / 6000);

    auto tree = DecisionTree(2, n_value);
    tree.train(x_train, y_train, weight);
    auto predicted = tree.predict(x_validation);
    int correct = 0;
    for (int i = 0; i < y_validation.size(); i++) {
        if (predicted[i] == y_validation[i]) {
            correct++;
        }
    }
    std::cout << "single: " << static_cast<double>(correct) / y_validation.size() << std::endl;
}

void mnist_tree_main() {
    auto data_x = MNISTUtility::read_images("/Users/chuxiaomin/Develop/ai/ml/train-images-idx3-ubyte");
    auto data_y = MNISTUtility::read_labels("/Users/chuxiaomin/Develop/ai/ml/train-labels-idx1-ubyte");
    xt::xarray<int> x_whole = MNISTUtility::pooled_binary_split(data_x);
    x_whole.reshape({60000, 14 * 14});

    xt::xarray<int> x_train = xt::view(x_whole, xt::range(_, 50000), xt::all());
    xt::xarray<int> x_validation = xt::view(x_whole, xt::range(50000, _), xt::all());
    auto y_train = std::vector<int>(data_y.begin(), data_y.begin() + 50000);
    auto y_validation = std::vector<int>(data_y.begin() + 50000, data_y.end());

    xt::xarray<double> weight = xt::ones<double>({50000}) * (1.0 / 50000);
    xt::xarray<double> validation_weight = xt::ones<double>({10000}) * (1.0 / 10000);

    std::vector<int> n_value(14 * 14, 2);

    std::vector<std::unique_ptr<WeightedClassifier>> weaks;
    constexpr int n_boost = 100;
    weaks.reserve(n_boost);
    for (int i = 0; i < n_boost; i++) {
        weaks.push_back(std::unique_ptr<WeightedClassifier>(new DecisionTree(10, n_value)));
    }
    auto boosted = AdaBoost(10, x_train, y_train, weight, weaks, x_validation, y_validation, validation_weight);
}

int main() {
    auto x_train_raw = xt::load_npy<double>("/Users/chuxiaomin/Develop/ai/ml/data/SST-2/x_train.npy");
    auto x_test = xt::load_npy<double>("/Users/chuxiaomin/Develop/ai/ml/data/SST-2/x_test.npy");
    auto y_train_xt = xt::load_npy<int>("/Users/chuxiaomin/Develop/ai/ml/data/SST-2/y_train.npy");
    auto y_test_xt = xt::load_npy<int>("/Users/chuxiaomin/Develop/ai/ml/data/SST-2/y_test.npy");
    auto y_train_raw = std::vector<int>(y_train_xt.begin(), y_train_xt.end());
    auto y_test = std::vector<int>(y_test_xt.begin(), y_test_xt.end());

    int train_size = y_train_raw.size() * 0.8;
    int validation_size = y_train_raw.size() - train_size;

    auto x_train = xt::view(x_train_raw, xt::range(_, train_size), xt::all());
    auto x_validation = xt::view(x_train_raw, xt::range(train_size, _), xt::all());
    auto y_train = std::vector<int>(y_train_raw.begin(), y_train_raw.begin() + train_size);
    auto y_validation = std::vector<int>(y_train_raw.begin() + train_size, y_train_raw.end());

    auto classifier = KNN(2, x_train, y_train);
    auto ks = std::vector<int>({1, 2, 3, 4, 5});
    auto predicted = classifier.predict(x_validation, ks);

    for (int i = 0; i < ks.size(); i++) {
        int correct = 0;
        for (int j = 0; j < validation_size; j++) {
            if (predicted[i][j] == y_validation[j]) {
                correct++;
            }
        }
        std::cout << " k: " << ks[i] << ", accuracy: " << static_cast<double>(correct) / validation_size << std::endl;
    }
}
