package com.cscao.apps.mobirnn;

import static com.cscao.apps.mobirnn.DataUtil.parseBias;
import static com.cscao.apps.mobirnn.DataUtil.parseWeight;
import static com.cscao.apps.mobirnn.DataUtil.sigmod;
import static com.cscao.apps.mobirnn.DataUtil.tanh;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * LSTM Model class
 */

public class Model {

    private float[][][] weights;
    private float[][] biases;
    private float[][] w_in;
    private float[][] w_out;
    private float[] b_in;
    private float[] b_out;

    public Model() {
    }

    public Model(String dataFolder) throws IOException {
        String bInPath = dataFolder + "b_in.csv";
        String bOutPath = dataFolder + "b_out.csv";
        String wInPath = dataFolder + "w_in.csv";
        String wOutPath = dataFolder + "w_out.csv";

        String[] weightsPath = new File(dataFolder).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith("weights");
            }
        });

        String[] biasesPath = new File(dataFolder).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith("biases");
            }
        });

        int layerSize = weightsPath.length;
        if (layerSize != biasesPath.length) throw new AssertionError();

        float[] b_in = parseBias(bInPath);
        float[] b_out = parseBias(bOutPath);

        float[][] w_in = parseWeight(wInPath);
        float[][] w_out = parseWeight(wOutPath);

        float[][][] weights = new float[layerSize][][];
        for (int i = 0; i < layerSize; i++) {
            weights[i] = parseWeight(weightsPath[i]);
        }

        float[][] biases = new float[layerSize][];
        for (int i = 0; i < layerSize; i++) {
            biases[i] = parseBias(biasesPath[i]);
        }

        this.weights = weights;
        this.biases = biases;
        this.w_in = w_in;
        this.w_out = w_out;
        this.b_in = b_in;
        this.b_out = b_out;
    }

    private float[][][] calcCellOneStep(float[][] in_, float[][] c_, float[][] h_, int layer) {
        // concat in_ and h_ and do xw_plu_b
        float[][] concat = Matrix.concat(in_, h_);
        float[][] linearResult = Matrix.addVec(Matrix.multiply(concat, this.weights[layer]),
                this.biases[layer]);
        float[][] i = Matrix.split(linearResult, 4, 0);
        float[][] j = Matrix.split(linearResult, 4, 1);
        float[][] f = Matrix.split(linearResult, 4, 2);
        float[][] o = Matrix.split(linearResult, 4, 3);


        int colSize = c_.length;
        int rowSize = c_[0].length;
        for (int k = 0; k < colSize; k++) {
            for (int l = 0; l < rowSize; l++) {
                c_[k][l] = c_[k][l] * sigmod(f[k][l] + 1) + sigmod(i[k][l]) * tanh(j[k][l]);
                h_[k][l] = tanh(c_[k][l]) * sigmod(o[k][l]);
            }
        }

        float[][][] state = new float[2][][];
        state[0] = c_;
        state[1] = h_;
        return state;
    }

    public int[] predict(float[][][] x) {
        int[] labels = new int[x.length];


        return labels;
    }
}
