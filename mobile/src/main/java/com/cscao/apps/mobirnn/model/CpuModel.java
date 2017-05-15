package com.cscao.apps.mobirnn.model;

import static com.cscao.apps.mobirnn.helper.DataUtil.alter2Dto1D;

import android.content.Context;

import com.cscao.apps.mobirnn.helper.DataUtil;
import com.cscao.apps.mobirnn.helper.Matrix;

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

    private ModelMode mMode;

    CpuModel(Context context, int layerSize, int hiddenUnits, ModelMode mode) {
        super(context, layerSize, hiddenUnits);
        mMode = mode;
        transformParams();
    }

    private void transformParams() {

        mWIn = new float[mInputDim][mHiddenUnits];
        for (int i = 0; i < mInputDim; i++) {
            System.arraycopy(super.mWIn, i * mHiddenUnits, mWIn[i], 0, mHiddenUnits);
        }


        mWOut = new float[mHiddenUnits][mOutputDim];
        for (int i = 0; i < mHiddenUnits; i++) {
            System.arraycopy(super.mWOut, i * mOutputDim, mWOut[i], 0, mOutputDim);
        }

        mWeights = new float[mLayerSize][mHiddenUnits * 2][mHiddenUnits * 4];
        for (int k = 0; k < mLayerSize; k++) {
            for (int i = 0; i < mHiddenUnits * 2; i++) {
                System.arraycopy(super.mRnnWeights[k], i * mHiddenUnits * 4, mWeights[k][i], 0,
                        mHiddenUnits * 4);
            }
        }
    }

    private float[][] calcCellOneStep(float[] in_, float[] c_, float[] h_, int layer) {
        // concat in_ and h_ and do xw_plu_b
        float[] concat = Matrix.concat(in_, h_);
        float[] linearResult = Matrix.vecAddVec(Matrix.vecMulMat(concat, mWeights[layer]),
                mRnnBiases[layer]);
        float[] i = Matrix.split(linearResult, 4, 0);
        float[] j = Matrix.split(linearResult, 4, 1);
        float[] f = Matrix.split(linearResult, 4, 2);
        float[] o = Matrix.split(linearResult, 4, 3);

        int size = c_.length;
        for (int k = 0; k < size; k++) {
            c_[k] = c_[k] * DataUtil.sigmoid(f[k] + 1) + DataUtil.sigmoid(i[k]) * DataUtil.tanh(
                    j[k]);
            h_[k] = DataUtil.tanh(c_[k]) * DataUtil.sigmoid(o[k]);
        }

        float[][] state = new float[2][];
        state[0] = c_;
        state[1] = h_;
        return state;
    }

    @Override
    protected int predictLabel(float[] x) {
        switch (mMode) {
            case CPU:
                return predictJavaCpu(x);
            case Native:
                return predictNativeCpu(x);
            case Eigen:
                return predictEigenCpu(x);
            default:
                return predictJavaCpu(x);
        }
    }

    private int predictJavaCpu(float[] input) {
        float[][] x = new float[mTimeSteps][mInputDim];
        for (int i = 0; i < mTimeSteps; i++) {
            System.arraycopy(input, i * mInputDim, x[i], 0, mInputDim);
        }

        x = DataUtil.relu(Matrix.addVec(Matrix.multiply(x, mWIn), mBIn));
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
            }
        }

        float[] outProb = Matrix.vecAddVec(Matrix.vecMulMat(h, mWOut), mBOut);
        return DataUtil.argmax(outProb) + 1;
    }

    private int predictNativeCpu(float[] x) {
        return predictNative(x, mLayerSize, mTimeSteps, mHiddenUnits, mInputDim, mOutputDim,
                super.mWIn, mBIn, super.mWOut, mBOut, alter2Dto1D(super.mRnnWeights),
                alter2Dto1D(mRnnBiases));
    }

    private int predictEigenCpu(float[] x) {
        return predictNativeEigen(x,
                new int[]{mLayerSize, mTimeSteps, mHiddenUnits, mInputDim, mOutputDim}, super.mWIn,
                mBIn, super.mWOut, mBOut, alter2Dto1D(super.mRnnWeights), alter2Dto1D(mRnnBiases));

    }

    private native int predictNativeEigen(float[] input, int[] config, float[] wIn, float[] bIn,
            float[] wOut, float[] bOut, float[] weights, float[] biases);

    private native int predictNative(float[] input, int layer_size,
            int time_steps, int hidden_unites, int in_dim, int out_dim, float[] wIn, float[] bIn,
            float[] wOut, float[] bOut, float[] weights, float[] biases);
}
