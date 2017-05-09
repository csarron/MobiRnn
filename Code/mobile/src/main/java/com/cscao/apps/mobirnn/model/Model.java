package com.cscao.apps.mobirnn.model;

import static com.cscao.apps.mobirnn.model.DataUtil.alter2Dto1D;
import static com.cscao.apps.mobirnn.model.DataUtil.alter3Dto1D;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import com.cscao.apps.mobirnn.ScriptC_main;
import com.orhanobut.logger.Logger;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;

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
    private int layerSize;
    private int hidden_units;
    private Model.MODE mMODE = MODE.GPU;
    private RenderScript mRs;
    private float[] convertedWIn;
    private float[] convertedWOut;
    private float[] convertedWeights;
    private float[] convertedBiases;
    private int inDim;
    private int outDim;
    private ScriptC_main scriptC_main;

    static {
        System.loadLibrary("main");
    }

    public enum MODE {
        CPU("CPU"), NATIVE("Native"), GPU("GPU");
        String mType;

        MODE(String type) {
            mType = type;
        }

        @Override
        public String toString() {
            return mType;
        }
    }

    public Model(String modelFolder, Model.MODE mode) throws IOException {
        this(modelFolder);
        this.mMODE = mode;
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

        float[] b_in = DataUtil.parseBias(bInPath);
        float[] b_out = DataUtil.parseBias(bOutPath);
        this.hidden_units = b_in.length;

        float[][] w_in = DataUtil.parseWeight(wInPath);
        float[][] w_out = DataUtil.parseWeight(wOutPath);

        float[][][] weights = new float[layerSize][][];
        for (int i = 0; i < layerSize; i++) {
            weights[i] = DataUtil.parseWeight(dataPath + weightsPath[i]);
        }

        float[][] biases = new float[layerSize][];
        for (int i = 0; i < layerSize; i++) {
            biases[i] = DataUtil.parseBias(dataPath + biasesPath[i]);
        }

        this.weights = weights;
        this.biases = biases;
        this.w_in = w_in;
        this.w_out = w_out;
        this.b_in = b_in;
        this.b_out = b_out;
        convertedWIn = alter2Dto1D(w_in);
        convertedWOut = alter2Dto1D(w_out);
        convertedWeights = alter3Dto1D(weights);
        convertedBiases = alter2Dto1D(biases);
        inDim = w_in.length;
        outDim = b_out.length;
    }

    private float[][] calcCellOneStep(float[] in_, float[] c_, float[] h_, int layer) {
        // concat in_ and h_ and do xw_plu_b
        float[] concat = Matrix.concat(in_, h_);
        float[] linearResult = Matrix.vecAddVec(Matrix.vecMulMat(concat, this.weights[layer]),
                this.biases[layer]);
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

    public int predict(float[][] x) {
        switch (mMODE) {
            case CPU:
                return predictOnCpu(x);
            case GPU:
                return predictOnGpu(x);
            case NATIVE:
                return predictTF(x);
            default:
                return predictOnGpu(x);
        }
    }

    private int predictNativeCpu(float[][] x) {
        int timeSteps = x.length;
        float[] inputs = alter2Dto1D(x);
        return predictNative(layerSize, timeSteps, hidden_units, inDim, outDim, convertedWIn, b_in,
                convertedWOut, b_out, convertedWeights, convertedBiases, inputs);

    }

    private int predictOnCpu(float[][] x) {
//        int timeSteps = x.length;

//        float[][] outputs = new float[timeSteps][];
        x = DataUtil.relu(Matrix.addVec(Matrix.multiply(x, w_in), b_in));
//        try {
//            for (float[] aX : x) {
//                File xFile = new File(getDataPath() + File.separator + "x_relu.log");
//                FileUtils.write(xFile, Arrays.toString(aX) + "\n", true);
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        float[] c = new float[hidden_units];
        float[] h = new float[hidden_units];
        for (int j = 0; j < layerSize; j++) {
            Arrays.fill(c, 0);
            Arrays.fill(h, 0);
            for (float[] aX : x) {
                float[][] state = calcCellOneStep(aX, c, h, j);
                c = state[0];
                h = state[1];
                System.arraycopy(h, 0, aX, 0, hidden_units);
//                outputs[k] = new float[hidden_units];
//                System.arraycopy(h, 0, outputs[k], 0, hidden_units);
//                x[k] = h;
//                outputs[k] = h;
            }
        }

        float[] outProb = Matrix.vecAddVec(Matrix.vecMulMat(h, w_out), b_out);
//        System.out.println("out:" + Arrays.toString(outProb).replaceAll("[\\[ | \\] | ,]", " "));
        return DataUtil.argmax(outProb) + 1;
    }

    private int predictOnGpu(float[][] x) {
        if (mRs == null) {
            return -1;
        }

        int timeSteps = x.length;
        float[] convertedX = alter2Dto1D(x);

        scriptC_main.set_time_steps(timeSteps);
        scriptC_main.set_in_dim(inDim);
        scriptC_main.set_hidden_unites(hidden_units);
        scriptC_main.set_layer_size(layerSize);

        // initialize input raw data allocation
        Type inRawType = Type.createXY(mRs, Element.F32(mRs), inDim, timeSteps);
        Allocation inputRawAlloc = Allocation.createTyped(mRs, inRawType);
        inputRawAlloc.copyFrom(convertedX);
        scriptC_main.set_input_raw(inputRawAlloc);

        // initialize activated input data allocation
        Type cellDataType = Type.createXY(mRs, Element.F32(mRs), hidden_units, timeSteps);
        Allocation inputsAlloc = Allocation.createTyped(mRs, cellDataType);
        scriptC_main.set_inputs(inputsAlloc);

        // initialize model parameters(weights and biases) allocation
        initializeParamsAllocation();

        allocIntermediateVariables();

        // initialize label probability output allocation
        Allocation labelProbAlloc = Allocation.createSized(mRs, Element.F32(mRs), outDim);
        scriptC_main.bind_label_prob(labelProbAlloc);
        scriptC_main.set_out_dim(outDim);

        // begin model forward pass computation
        long start = System.currentTimeMillis();
        scriptC_main.invoke_all_in_one();
////        scriptC_main.forEach_input_transform(inputsAlloc);
//        scriptC_main.invoke_input_transform_func();
//        for (int i = 0; i < layerSize; i++) {
////            scriptC_main.forEach_set_zeros(cAlloc);
////            scriptC_main.forEach_set_zeros(hAlloc);
//            scriptC_main.invoke_set_ch_zeros();
//            scriptC_main.set_current_layer(i);
//            for (int j = 0; j < timeSteps; j++) {
//                scriptC_main.set_current_step(j);
//                scriptC_main.invoke_calc_cell_one_step();
////                scriptC_main.invoke_concat_in_h();
//
////                scriptC_main.forEach_linear_map(linearResultAlloc);
////                scriptC_main.invoke_linear_map_func();
//
////                scriptC_main.forEach_pointwise_ch(cAlloc);// or pass hAlloc
////                scriptC_main.invoke_pointwise_ch_func();
//
////                scriptC_main.forEach_update_input(hAlloc);
////                scriptC_main.invoke_update_input_func();
//            }
//        }
////        scriptC_main.forEach_output_transform(labelProbAlloc);
//        scriptC_main.invoke_output_transform_func();
        mRs.finish();

        long end = System.currentTimeMillis();

        // copy result back
        float[] labelProb = new float[outDim];
        labelProbAlloc.copyTo(labelProb);
        Logger.i("invoke time: %s", (end - start));
        return DataUtil.argmax(labelProb) + 1;
    }

    private void initializeParamsAllocation() {
        Type wInType = Type.createXY(mRs, Element.F32(mRs), hidden_units, inDim);
        Allocation wInAlloc = Allocation.createTyped(mRs, wInType);
        wInAlloc.copyFrom(convertedWIn);
        scriptC_main.set_w_in(wInAlloc);

        Allocation bInAlloc = Allocation.createSized(mRs, Element.F32(mRs), hidden_units);
        bInAlloc.copyFrom(b_in);
        scriptC_main.set_b_in(bInAlloc);

        Type wOutType = Type.createXY(mRs, Element.F32(mRs), outDim, hidden_units);
        Allocation wOutAlloc = Allocation.createTyped(mRs, wOutType);
        wOutAlloc.copyFrom(convertedWOut);
        scriptC_main.set_w_out(wOutAlloc);

        Allocation bOutAlloc = Allocation.createSized(mRs, Element.F32(mRs), outDim);
        bOutAlloc.copyFrom(b_out);
        scriptC_main.set_b_out(bOutAlloc);

        Type weightType = Type.createXYZ(mRs, Element.F32(mRs),
                hidden_units * 4, hidden_units * 2, layerSize);
        Allocation weightAlloc = Allocation.createTyped(mRs, weightType);
        weightAlloc.copyFrom(convertedWeights);
        scriptC_main.set_weights(weightAlloc);

        Type biasType = Type.createXY(mRs, Element.F32(mRs), hidden_units * 4, layerSize);
        Allocation biasAlloc = Allocation.createTyped(mRs, biasType);
        biasAlloc.copyFrom(convertedBiases);
        scriptC_main.set_biases(biasAlloc);
    }

    private void allocIntermediateVariables() {
        Allocation cAlloc = Allocation.createSized(mRs, Element.F32(mRs), hidden_units);
        scriptC_main.bind_c(cAlloc);

        Allocation hAlloc = Allocation.createSized(mRs, Element.F32(mRs), hidden_units);
        scriptC_main.bind_h(hAlloc);

        Allocation inputConcatAlloc = Allocation.createSized(mRs, Element.F32(mRs),
                hidden_units * 2);
        scriptC_main.bind_input_concat(inputConcatAlloc);

        Allocation linearResultAlloc = Allocation.createSized(mRs, Element.F32(mRs),
                hidden_units * 4);
        scriptC_main.bind_linear_result(linearResultAlloc);

    }

    public void setRs(RenderScript rs) {
        mRs = rs;
        scriptC_main = new ScriptC_main(rs);
    }

    public native int predictNative(int layerSize, int timeSteps, int hiddenUnits, int inDim,
            int outDim,
            float[] convertedWIn, float[] bIn, float[] convertedWOut,
            float[] bOut, float[] convertedWeights, float[] convertedBiases, float[] input);

    private TensorFlowInferenceInterface mInferenceInterface;

    public void setInferenceInterface(
            TensorFlowInferenceInterface inferenceInterface) {
        mInferenceInterface = inferenceInterface;
    }

    private int predictTF(float[][] x) {
        int timeSteps = x.length;
        float[] convertedX = alter2Dto1D(x);
        mInferenceInterface.feed("input", convertedX, 1, inDim, timeSteps);
        mInferenceInterface.run(new String[]{"output"});
        float[] labelProb = new float[outDim];
        mInferenceInterface.fetch("output", labelProb);

        return DataUtil.argmax(labelProb) + 1;
    }
}
