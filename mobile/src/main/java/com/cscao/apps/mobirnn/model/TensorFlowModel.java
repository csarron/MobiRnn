package com.cscao.apps.mobirnn.model;

import static com.cscao.apps.mobirnn.helper.DataUtil.alter2Dto1D;

import android.content.Context;

import com.cscao.apps.mobirnn.helper.DataUtil;

/**
 * Created by qqcao on 5/11/17Thursday.
 *
 * LSTM model using TensorFlow
 */

class TensorFlowModel extends AbstractModel {
    private static final String mInputNodeName = "input";
    private static final String mOutputNodeName = "output";

    TensorFlowModel(Context context, int layerSize, int hiddenUnits) {
        super(context, layerSize, hiddenUnits);
    }

    @Override
    protected int predictLabel(float[][] x) {
        int timeSteps = x.length;
        float[] convertedX = alter2Dto1D(x);
        mTensorFlowInferenceInterface.feed(mInputNodeName, convertedX, 1, mInputDim, timeSteps);
        mTensorFlowInferenceInterface.run(new String[]{mOutputNodeName});
        float[] labelProb = new float[mOutputDim];
        mTensorFlowInferenceInterface.fetch(mOutputNodeName, labelProb);
        return DataUtil.argmax(labelProb) + 1;
    }
}
