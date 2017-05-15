package com.cscao.apps.mobirnn.model;

public enum ModelMode {
    CPU(0), Native(1), GPU(2), TensorFlow(3), Eigen(4);
    private int mType;
    ModelMode(int type) {
        mType = type;
    }
    private static final String[] MODES = {"CPU", "Native", "GPU", "TensorFlow", "Eigen"};
    @Override
    public String toString() {
        return MODES[mType];
    }
}