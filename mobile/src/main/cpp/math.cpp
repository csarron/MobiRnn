//
// Created by Qingqing Cao on 4/13/17Thursday.
//
#include "math.h"
float* addVec(float* m, int mx, int my, float* v, int vy){
    if (my != vy) {
        return NULL;
    }

    float* r = (float*) malloc( mx * my * sizeof(float));
    for (int i = 0; i < mx; i++) {
        for (int j = 0; j < my; j++) {
            *(r + i*mx + j) = *(m + i*mx + j) + v[j];
        }
    }
    return r;
}

float* vecAddVec(float* a, float* b, int len){
    float* c = (float*) malloc( len * sizeof(float));
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

float* multiply(float* a, float* b, int m, int p, int n){
    float* c = (float*) malloc( m * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++) {
                *(c + i*m + j) += *(a + i*m + k) * *(b + k*p + j);
            }
        }
    }
    return c;
}

float sigmoid(float x) {
    return 1 / (1 + (float) exp(-x));
}