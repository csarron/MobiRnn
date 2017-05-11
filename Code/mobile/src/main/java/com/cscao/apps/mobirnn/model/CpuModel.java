package com.cscao.apps.mobirnn.model;

import android.content.Context;

import com.cscao.apps.mobirnn.helper.DataUtil;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Created by qqcao on 5/11/17Thursday.
 *
 * LSTM model on CPU
 */

class CpuModel extends AbstractModel {
    static {
        System.loadLibrary("main");
    }

    private float[][] mWIn;
    private float[][] mWOut;
    private float[][][] mWeights;
    private float[][] mBiases;

    CpuModel(Context context, int layerSize, int hiddenUnits) {
        super(context, layerSize, hiddenUnits);
        transformParams();
    }

    private void transformParams() {
        INDArray win = Nd4j.create(super.mWIn);
        win = win.reshape(mInputDim, 4 * mHiddenUnits);
        mWIn = new float[mInputDim][4 * mHiddenUnits];
        for (int i = 0; i < win.rows(); i++) {
            for (int j = 0; j < win.columns(); j++) {
                mWIn[i][j] = win.getFloat(i, j);
            }
        }

        INDArray wout = Nd4j.create(super.mWOut);
        wout = wout.reshape(mHiddenUnits, mOutputDim);
        mWOut = new float[mHiddenUnits][mOutputDim];
        for (int i = 0; i < wout.rows(); i++) {
            for (int j = 0; j < wout.columns(); j++) {
                mWOut[i][j] = wout.getFloat(i, j);
            }
        }

        INDArray weights = Nd4j.create(super.mRnnWeights);
        weights = weights.reshape(mLayerSize, mHiddenUnits * 2, mHiddenUnits * 4);
        mWeights = new float[mLayerSize][mHiddenUnits * 2][mHiddenUnits * 4];
        for (int k = 0; k < mLayerSize; k++) {
            for (int i = 0; i < mHiddenUnits*2; i++) {
                for (int j = 0; j < mHiddenUnits*4; j++) {
                    mWeights[k][i][j] = weights.getFloat(new int[]{k, i, j});
                }
            }
        }
        INDArray biases = Nd4j.create(super.mRnnBiases);
        biases = biases.reshape(mLayerSize, 4 * mHiddenUnits);
        mBiases = new float[mLayerSize][4 * mHiddenUnits];
        for (int i = 0; i < biases.rows(); i++) {
            for (int j = 0; j < biases.columns(); j++) {
                mBiases[i][j] = biases.getFloat(i, j);
            }
        }
    }

    private float[][] calcCellOneStep(float[] in_, float[] c_, float[] h_, int layer) {
        // concat in_ and h_ and do xw_plu_b
        float[] concat = Matrix.concat(in_, h_);
        float[] linearResult = Matrix.vecAddVec(Matrix.vecMulMat(concat, mWeights[layer]),
                mBiases[layer]);
//        try {
//            File concatFile = new File(getDataPath() + File.separator + "concat.log");
//            FileUtils.write(concatFile, Arrays.toString(concat) + "\n", true);
//            File lRFile = new File(getDataPath() + File.separator + "linearResult.log");
//            FileUtils.write(lRFile, Arrays.toString(linearResult) + "\n", true);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        float[] i = Matrix.split(linearResult, 4, 0);
        float[] j = Matrix.split(linearResult, 4, 1);
        float[] f = Matrix.split(linearResult, 4, 2);
        float[] o = Matrix.split(linearResult, 4, 3);
//        try {
//            File i_File = new File(getDataPath() + File.separator + "i.log");
//            FileUtils.write(i_File, Arrays.toString(i) + "\n", true);
//            File j_File = new File(getDataPath() + File.separator + "j.log");
//            FileUtils.write(j_File, Arrays.toString(j) + "\n", true);
//            File fFile = new File(getDataPath() + File.separator + "f.log");
//            FileUtils.write(fFile, Arrays.toString(f) + "\n", true);
//            File oFile = new File(getDataPath() + File.separator + "o.log");
//            FileUtils.write(oFile, Arrays.toString(o) + "\n", true);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        int size = c_.length;
        for (int k = 0; k < size; k++) {
            c_[k] = c_[k] * DataUtil.sigmoid(f[k] + 1) + DataUtil.sigmoid(i[k]) * DataUtil.tanh(
                    j[k]);
            h_[k] = DataUtil.tanh(c_[k]) * DataUtil.sigmoid(o[k]);
        }
//        try {
//            File c_File = new File(getDataPath() + File.separator + "c_.log");
//            FileUtils.write(c_File, Arrays.toString(c_) + "\n", true);
//            File h_File = new File(getDataPath() + File.separator + "h_.log");
//            FileUtils.write(h_File, Arrays.toString(h_) + "\n", true);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        float[][] state = new float[2][];
        state[0] = c_;
        state[1] = h_;
        return state;
    }

    @Override
    protected int predictLabel(float[][] x) {
//        int timeSteps = x.length;

//        float[][] outputs = new float[timeSteps][];
        x = DataUtil.relu(Matrix.addVec(Matrix.multiply(x, mWIn), mBIn));
//        try {
//            for (float[] aX : x) {
//                File xFile = new File(getDataPath() + File.separator + "x_relu.log");
//                FileUtils.write(xFile, Arrays.toString(aX) + "\n", true);
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        float[] c = new float[mHiddenUnits];
        float[] h = new float[mHiddenUnits];
        for (int j = 0; j < mLayerSize; j++) {
            Arrays.fill(c, 0);
            Arrays.fill(h, 0);
            for (float[] aX : x) {
                float[][] state = calcCellOneStep(aX, c, h, j);
                c = state[0];
                h = state[1];
                System.arraycopy(h, 0, aX, 0, mHiddenUnits);
//                outputs[k] = new float[mHiddenUnits];
//                System.arraycopy(h, 0, outputs[k], 0, mHiddenUnits);
//                x[k] = h;
//                outputs[k] = h;
            }
        }

        float[] outProb = Matrix.vecAddVec(Matrix.vecMulMat(h, mWOut), mBOut);
//        System.out.println("out:" + Arrays.toString(outProb).replaceAll("[\\[ | \\] | ,]", " "));
        return DataUtil.argmax(outProb) + 1;
    }

    //    private int predictNativeCpu(float[][] x) {
//        int timeSteps = x.length;
//        float[] inputs = alter2Dto1D(x);
//        return predictNative(mLayerSize, timeSteps, mHiddenUnits, inDim, outDim, convertedWIn,
// b_in,
//                convertedWOut, b_out, convertedWeights, convertedBiases, inputs);
//
//    }
}
