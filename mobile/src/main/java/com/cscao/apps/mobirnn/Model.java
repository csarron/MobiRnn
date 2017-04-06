package com.cscao.apps.mobirnn;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * LSTM Model class
 */

public class Model {

    float[][][] weights;
    float[][] biases;
    float[][] w_in;
    float[][] w_out;
    float[] b_in;
    float[] b_out;

    public Model() {
    }

    public Model(float[][][] weights, float[][] biases, float[][] w_in, float[][] w_out,
            float[] b_in,
            float[] b_out) {
        this.weights = weights;
        this.biases = biases;
        this.w_in = w_in;
        this.w_out = w_out;
        this.b_in = b_in;
        this.b_out = b_out;
    }

    public Model(String dataFolder) {
        String bInPath = dataFolder + "b_in.csv";
        String bOutPath = dataFolder + "b_out.csv";
        String wInPath = dataFolder + "w_in.csv";
        String wOutPath = dataFolder + "w_out.csv";


    }
}
