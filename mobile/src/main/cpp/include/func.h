//
// Created by Qingqing Cao on 6/7/17.
//

#ifndef MOBIRNN_FUNC_H
#define MOBIRNN_FUNC_H
#include <jni.h>
#include <Eigen/Dense>
using namespace Eigen;

inline float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

inline ArrayXXf sigmoid(ArrayXXf arr) {
    arr *= -1;
    return (arr.exp() + 1).inverse();
}

inline int argMax(float *x, int len) {
    int max = 0;
    float a = 0.0f;
    for (int i = 0; i < len; i++) {
        if (x[i] > a) {
            a = x[i];
            max = i;
        }
    }
    return max;
}
#endif //MOBIRNN_FUNC_H
