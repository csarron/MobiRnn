package com.cscao.apps.mobirnn.model;

public enum ModelMode {
    CPU("CPU"), Native("Native"), GPU("GPU"), TensorFlow("TensorFlow");
    String mType;

    ModelMode(String type) {
        mType = type;
    }

    @Override
    public String toString() {
        return mType;
    }
}