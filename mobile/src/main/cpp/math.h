//
// Created by Qingqing Cao on 4/13/17Thursday.
//

#ifndef MOBIRNN_MATH_H
#define MOBIRNN_MATH_H
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

float* addVec(float* m, int mx, int my, float* v, int vy);
float* vecAddVec(float* a, float* b, int len);
float* multiply(float* a, float* b, int m, int p, int n);
float sigmoid(float x);

#endif //MOBIRNN_MATH_H
