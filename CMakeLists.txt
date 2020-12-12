cmake_minimum_required(VERSION 3.17)
set(CMAKE_OSX_SYSROOT /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.0.sdk)
project(MLProject)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH /usr/local/Caskroom/miniconda/base;${CMAKE_PREFIX_PATH})

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtl REQUIRED)
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

add_definitions(-DHAVE_CBLAS=1)
add_definitions(-DXTENSOR_USE_XSIMD)

add_executable(MLProject main.cpp Classifier.cpp Classifier.h DecisionTree.cpp DecisionTree.h KNNMNISTClassifier.cpp KNNMNISTClassifier.h Utility.h Utility.cpp AdaBoost.cpp AdaBoost.h WeightedClassifier.h NaiveBayes.cpp NaiveBayes.h)

target_include_directories(MLProject PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(MLProject PUBLIC xtensor xtensor::optimize xtensor::use_xsimd ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
