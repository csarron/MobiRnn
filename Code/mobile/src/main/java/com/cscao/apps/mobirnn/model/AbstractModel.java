package com.cscao.apps.mobirnn.model;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Locale;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * LSTM AbstractModel abstract class
 */

abstract class AbstractModel {

    private static final String[] mActivationWeightNames = /* order matters*/
            {"w_in", "b_in", "w_out", "b_out"};
    private static final String mModelNameTemplate = "%dlayer%dunits.pb";
    private static final String mRnnWeightTemplate =
            "rnn/multi_rnn_cell/cell_%d/basic_lstm_cell/weights";
    private static final String mRnnBiaseTemplate =
            "rnn/multi_rnn_cell/cell_%d/basic_lstm_cell/biases";
    final int mOutputDim = 6;
    final int mInputDim = 9;

    private String[] mRnnWeightNames;
    private String[] mRnnBiaseNames;

    Context mContext;
    TensorFlowInferenceInterface mTensorFlowInferenceInterface;

    int mLayerSize;
    int mHiddenUnits;

    float[][] mRnnWeights;
    float[][] mRnnBiases;

    float[] mWIn;
    float[] mBIn;
    float[] mWOut;
    float[] mBOut;

    AbstractModel(Context context, int layerSize, int hiddenUnits) {
        mContext = context;
        mLayerSize = layerSize;
        mHiddenUnits = hiddenUnits;
        init();
        loadModel();
    }

    private void init() {
        mTensorFlowInferenceInterface = new TensorFlowInferenceInterface(mContext.getAssets(),
                String.format(Locale.US, mModelNameTemplate, mLayerSize, mHiddenUnits));

        mRnnWeights = new float[mLayerSize][];
        mRnnBiases = new float[mLayerSize][];
        mRnnWeightNames = new String[mLayerSize];
        mRnnBiaseNames = new String[mLayerSize];
    }

    private void loadModel() {
        //prepareRnnWeightNames
        for (int i = 0; i < mLayerSize; i++) {
            mRnnWeightNames[i] = String.format(Locale.US, mRnnWeightTemplate, i);
            mRnnBiaseNames[i] = String.format(Locale.US, mRnnBiaseTemplate, i);
        }

        mTensorFlowInferenceInterface.run(mActivationWeightNames);

        mWIn = new float[mInputDim * mHiddenUnits];
        mBIn = new float[mHiddenUnits];
        mWOut = new float[mOutputDim * mHiddenUnits];
        mBOut = new float[mOutputDim];

        mTensorFlowInferenceInterface.fetch(mActivationWeightNames[0], mWIn);
        mTensorFlowInferenceInterface.fetch(mActivationWeightNames[1], mBIn);
        mTensorFlowInferenceInterface.fetch(mActivationWeightNames[2], mWOut);
        mTensorFlowInferenceInterface.fetch(mActivationWeightNames[3], mBOut);

        mTensorFlowInferenceInterface.run(mRnnWeightNames);
        for (int i = 0; i < mLayerSize; i++) {
            mRnnWeights[i] = new float[2 * mHiddenUnits * 4 * mHiddenUnits];
            mTensorFlowInferenceInterface.fetch(mRnnWeightNames[i], mRnnWeights[i]);
        }

        mTensorFlowInferenceInterface.run(mRnnBiaseNames);
        for (int i = 0; i < mLayerSize; i++) {
            mRnnBiases[i] = new float[4 * mHiddenUnits];
            mTensorFlowInferenceInterface.fetch(mRnnBiaseNames[i], mRnnBiases[i]);
        }
    }

    protected abstract int predictLabel(float[][] x);
}
