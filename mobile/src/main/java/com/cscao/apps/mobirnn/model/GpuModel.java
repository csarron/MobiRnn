package com.cscao.apps.mobirnn.model;

import static com.cscao.apps.mobirnn.helper.DataUtil.alter2Dto1D;

import android.content.Context;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import com.cscao.apps.mobirnn.ScriptC_main;
import com.cscao.apps.mobirnn.helper.DataUtil;
import com.orhanobut.logger.Logger;

/**
 * Created by qqcao on 5/11/17Thursday.
 *
 * LSTM model on GPU
 */

class GpuModel extends AbstractModel {
    private RenderScript mRs;
    private ScriptC_main scriptC_main;

    GpuModel(Context context, int layerSize, int hiddenUnits) {
        super(context, layerSize, hiddenUnits);
        mRs = RenderScript.create(mContext, RenderScript.ContextType.NORMAL);
        scriptC_main = new ScriptC_main(mRs);
        // initialize model parameters(weights and biases) allocation
        initializeParamsAllocation();
        allocIntermediateVariables();
    }

    @Override
    protected int predictLabel(float[] x) {
        if (mRs == null) {
            return -1;
        }

        scriptC_main.set_time_steps(mTimeSteps);
        scriptC_main.set_in_dim(mInputDim);
        scriptC_main.set_hidden_unites(mHiddenUnits);
        scriptC_main.set_layer_size(mLayerSize);

        // initialize input raw data allocation
        Type inRawType = Type.createXY(mRs, Element.F32(mRs), mInputDim, mTimeSteps);
        Allocation inputRawAlloc = Allocation.createTyped(mRs, inRawType);
        inputRawAlloc.copyFrom(x);
        scriptC_main.set_input_raw(inputRawAlloc);

        // initialize activated input data allocation
        Type cellDataType = Type.createXY(mRs, Element.F32(mRs), mHiddenUnits, mTimeSteps);
        Allocation inputsAlloc = Allocation.createTyped(mRs, cellDataType);
        scriptC_main.set_inputs(inputsAlloc);



        // initialize label probability output allocation
        Allocation labelProbAlloc = Allocation.createSized(mRs, Element.F32(mRs), mOutputDim);
        scriptC_main.bind_label_prob(labelProbAlloc);
        scriptC_main.set_out_dim(mOutputDim);

        // begin model forward pass computation
        long start = System.currentTimeMillis();
        scriptC_main.invoke_all_in_one();
        mRs.finish();

        long end = System.currentTimeMillis();

        // copy result back
        float[] labelProb = new float[mOutputDim];
        labelProbAlloc.copyTo(labelProb);
        Logger.i("invoke time: %s", (end - start));
        return DataUtil.argmax(labelProb) + 1;
    }

    private void initializeParamsAllocation() {
        Type wInType = Type.createXY(mRs, Element.F32(mRs), mHiddenUnits, mInputDim);
        Allocation wInAlloc = Allocation.createTyped(mRs, wInType);
        wInAlloc.copyFrom(mWIn);
        scriptC_main.set_w_in(wInAlloc);

        Allocation bInAlloc = Allocation.createSized(mRs, Element.F32(mRs), mHiddenUnits);
        bInAlloc.copyFrom(mBIn);
        scriptC_main.set_b_in(bInAlloc);

        Type wOutType = Type.createXY(mRs, Element.F32(mRs), mOutputDim, mHiddenUnits);
        Allocation wOutAlloc = Allocation.createTyped(mRs, wOutType);
        wOutAlloc.copyFrom(mWOut);
        scriptC_main.set_w_out(wOutAlloc);

        Allocation bOutAlloc = Allocation.createSized(mRs, Element.F32(mRs), mOutputDim);
        bOutAlloc.copyFrom(mBOut);
        scriptC_main.set_b_out(bOutAlloc);

        Type weightType = Type.createXYZ(mRs, Element.F32(mRs),
                mHiddenUnits * 4, mHiddenUnits * 2, mLayerSize);
        Allocation weightAlloc = Allocation.createTyped(mRs, weightType);
        weightAlloc.copyFrom(alter2Dto1D(mRnnWeights));
        scriptC_main.set_weights(weightAlloc);

        Type biasType = Type.createXY(mRs, Element.F32(mRs), mHiddenUnits * 4, mLayerSize);
        Allocation biasAlloc = Allocation.createTyped(mRs, biasType);
        biasAlloc.copyFrom(alter2Dto1D(mRnnBiases));
        scriptC_main.set_biases(biasAlloc);
    }

    private void allocIntermediateVariables() {
        Allocation cAlloc = Allocation.createSized(mRs, Element.F32(mRs), mHiddenUnits);
        scriptC_main.bind_c(cAlloc);

        Allocation hAlloc = Allocation.createSized(mRs, Element.F32(mRs), mHiddenUnits);
        scriptC_main.bind_h(hAlloc);

        Allocation inputConcatAlloc = Allocation.createSized(mRs, Element.F32(mRs),
                mHiddenUnits * 2);
        scriptC_main.bind_input_concat(inputConcatAlloc);

        Allocation linearResultAlloc = Allocation.createSized(mRs, Element.F32(mRs),
                mHiddenUnits * 4);
        scriptC_main.bind_linear_result(linearResultAlloc);

    }

}
