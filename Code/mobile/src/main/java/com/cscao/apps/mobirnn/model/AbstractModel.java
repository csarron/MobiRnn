package com.cscao.apps.mobirnn.model;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Locale;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * LSTM AbstractModel abstract class
 */

public abstract class AbstractModel {

    private static final String[] mActivationWeightNames = /* order matters*/
            {"w_in", "b_in", "w_out", "b_out"};
    private static final String mModelNameTemplate = "%d_layer_%d_units.pb";
    private static final String mRnnWeightTemplate =
            "rnn/multi_rnn_cell/cell_%d/basic_lstm_cell/weights";
    private static final String mRnnBiaseTemplate =
            "rnn/multi_rnn_cell/cell_%d/basic_lstm_cell/biases";
    private final int mActivationWeightSize = mActivationWeightNames.length;

    private String[] mRnnWeightNames;
    private String[] mRnnBiaseNames;

    Context mContext;
    TensorFlowInferenceInterface mTensorFlowInferenceInterface;

    int mLayerSize;
    int mHiddenUnits;

    private float[][] mActivationParams;
    float[][] mRnnWeights;
    float[][] mRnnBiases;

    int mInputDim;
    int mOutputDim;

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

        mActivationParams = new float[mActivationWeightSize][];
        mRnnWeights = new float[mLayerSize][];
        mRnnBiases = new float[mLayerSize][];
        mRnnWeightNames = new String[mLayerSize];
        mRnnBiaseNames = new String[mLayerSize];
    }

    private void loadModel() {
        //prepareRnnWeightNames
        for (int i = 0; i < mLayerSize; i++) {
            mRnnWeightNames[i] = String.format(Locale.US, mRnnWeightTemplate, mLayerSize);
            mRnnBiaseNames[i] = String.format(Locale.US, mRnnBiaseTemplate, mLayerSize);
        }

        //fake input data
        float[] convertedX = new float[mInputDim*128];
        mTensorFlowInferenceInterface.feed("input", convertedX, 1, mInputDim, 128);

        mTensorFlowInferenceInterface.run(mActivationWeightNames);
        for (int i = 0; i < mActivationWeightSize; i++) {
            mTensorFlowInferenceInterface.fetch(mActivationWeightNames[i], mActivationParams[i]);
        }

        mTensorFlowInferenceInterface.run(mRnnWeightNames);
        mTensorFlowInferenceInterface.run(mRnnBiaseNames);
        for (int i = 0; i < mLayerSize; i++) {
            mTensorFlowInferenceInterface.fetch(mRnnWeightNames[i], mRnnWeights[i]);
            mTensorFlowInferenceInterface.fetch(mRnnBiaseNames[i], mRnnBiases[i]);
        }

        mInputDim = mActivationParams[0].length; // input dimension is w_in 1st dimension
        mOutputDim = mActivationParams[3].length; // output dimension is b_out length

        mWIn = mActivationParams[0];
        mBIn = mActivationParams[1];
        mWOut = mActivationParams[2];
        mBOut = mActivationParams[3];
    }

    protected abstract int predictLabel(float[][] x);
}
