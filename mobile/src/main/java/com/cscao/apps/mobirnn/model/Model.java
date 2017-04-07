package com.cscao.apps.mobirnn.model;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * LSTM Model class
 */

public class Model {

    private double[][][] weights;
    private double[][] biases;
    private double[][] w_in;
    private double[][] w_out;
    private double[] b_in;
    private double[] b_out;
    private int layerSize;
    private int hidden_units;

    public Model() {
    }

    public Model(String dataFolder) throws IOException {
        String dataPath = dataFolder + File.separator;
        String bInPath = dataPath + "b_in.csv";
        String bOutPath = dataPath + "b_out.csv";
        String wInPath = dataPath + "w_in.csv";
        String wOutPath = dataPath + "w_out.csv";

        String[] weightsPath = new File(dataPath).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith("weights.csv");
            }
        });

        String[] biasesPath = new File(dataPath).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith("biases.csv");
            }
        });

        int layerSize = weightsPath.length;
        if (layerSize != biasesPath.length) throw new AssertionError();
        this.layerSize = layerSize;

        double[] b_in = DataUtil.parseBias(bInPath);
        double[] b_out = DataUtil.parseBias(bOutPath);
        this.hidden_units = b_in.length;

        double[][] w_in = DataUtil.parseWeight(wInPath);
        double[][] w_out = DataUtil.parseWeight(wOutPath);

        double[][][] weights = new double[layerSize][][];
        for (int i = 0; i < layerSize; i++) {
            weights[i] = DataUtil.parseWeight(dataPath + weightsPath[i]);
        }

        double[][] biases = new double[layerSize][];
        for (int i = 0; i < layerSize; i++) {
            biases[i] = DataUtil.parseBias(dataPath + biasesPath[i]);
        }

        this.weights = weights;
        this.biases = biases;
        this.w_in = w_in;
        this.w_out = w_out;
        this.b_in = b_in;
        this.b_out = b_out;
    }

    private double[][] calcCellOneStep(double[] in_, double[] c_, double[] h_, int layer) {
        // concat in_ and h_ and do xw_plu_b
        double[] concat = Matrix.concat(in_, h_);
        double[] linearResult = Matrix.vecAddVec(Matrix.vecMulMat(concat, this.weights[layer]),
                this.biases[layer]);
        double[] i = Matrix.split(linearResult, 4, 0);
        double[] j = Matrix.split(linearResult, 4, 1);
        double[] f = Matrix.split(linearResult, 4, 2);
        double[] o = Matrix.split(linearResult, 4, 3);

        int size = c_.length;
        for (int k = 0; k < size; k++) {
            c_[k] = c_[k] * DataUtil.sigmod(f[k] + 1) + DataUtil.sigmod(i[k]) * DataUtil.tanh(j[k]);
            h_[k] = DataUtil.tanh(c_[k]) * DataUtil.sigmod(o[k]);
        }

        double[][] state = new double[2][];
        state[0] = c_;
        state[1] = h_;
        return state;
    }

    public int predict(double[][] x) {
        int timeSteps = x.length;

        double[][] outputs = new double[timeSteps][];
        x = DataUtil.relu(Matrix.addVec(Matrix.multiply(x, w_in), b_in));

        for (int j = 0; j < layerSize; j++) {
            double[] c = new double[hidden_units];
            double[] h = new double[hidden_units];

            for (int k = 0; k < timeSteps; k++) {
                double[][] state = calcCellOneStep(x[k], c, h, j);
                c = state[0];
                h = state[1];
                System.arraycopy(h, 0, x[k], 0, hidden_units);
                outputs[k] = new double[hidden_units];
                System.arraycopy(h, 0, outputs[k], 0, hidden_units);
//                x[k] = h;
//                outputs[k] = h;
            }
        }

        double[] outProb = Matrix.vecAddVec(Matrix.vecMulMat(outputs[timeSteps - 1], w_out), b_out);
//        System.out.println("out:" + Arrays.toString(outProb).replaceAll("[\\[ | \\] | ,]", " "));
        return DataUtil.argmax(outProb) + 1;
    }
}
