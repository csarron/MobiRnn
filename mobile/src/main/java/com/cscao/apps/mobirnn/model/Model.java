package com.cscao.apps.mobirnn.model;

import static com.cscao.apps.mobirnn.model.DataUtil.alter2Dto1D;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import com.cscao.apps.mobirnn.ScriptC_main;

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
    private boolean isModeCpu = true;
    private RenderScript mRs;

    public Model(String modelFolder, boolean isModeCpu) throws IOException {
        this(modelFolder);
        this.isModeCpu = isModeCpu;
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
    }

    private float[][] calcCellOneStep(float[] in_, float[] c_, float[] h_, int layer) {
        // concat in_ and h_ and do xw_plu_b
        float[] concat = Matrix.concat(in_, h_);
        float[] linearResult = Matrix.vecAddVec(Matrix.vecMulMat(concat, this.weights[layer]),
                this.biases[layer]);
        float[] i = Matrix.split(linearResult, 4, 0);
        float[] j = Matrix.split(linearResult, 4, 1);
        float[] f = Matrix.split(linearResult, 4, 2);
        float[] o = Matrix.split(linearResult, 4, 3);

        int size = c_.length;
        for (int k = 0; k < size; k++) {
            c_[k] = c_[k] * DataUtil.sigmod(f[k] + 1) + DataUtil.sigmod(i[k]) * DataUtil.tanh(j[k]);
            h_[k] = DataUtil.tanh(c_[k]) * DataUtil.sigmod(o[k]);
        }

        float[][] state = new float[2][];
        state[0] = c_;
        state[1] = h_;
        return state;
    }

    public int predict(float[][] x) {
        if (isModeCpu) {
            return predictOnCpu(x);
        } else {
            return predictOnGpu(x);
        }
    }

    private int predictOnCpu(float[][] x) {
        int timeSteps = x.length;

        float[][] outputs = new float[timeSteps][];
        x = DataUtil.relu(Matrix.addVec(Matrix.multiply(x, w_in), b_in));

        for (int j = 0; j < layerSize; j++) {
            float[] c = new float[hidden_units];
            float[] h = new float[hidden_units];

            for (int k = 0; k < timeSteps; k++) {
                float[][] state = calcCellOneStep(x[k], c, h, j);
                c = state[0];
                h = state[1];
                System.arraycopy(h, 0, x[k], 0, hidden_units);
                outputs[k] = new float[hidden_units];
                System.arraycopy(h, 0, outputs[k], 0, hidden_units);
//                x[k] = h;
//                outputs[k] = h;
            }
        }

        float[] outProb = Matrix.vecAddVec(Matrix.vecMulMat(outputs[timeSteps - 1], w_out), b_out);
//        System.out.println("out:" + Arrays.toString(outProb).replaceAll("[\\[ | \\] | ,]", " "));
        return DataUtil.argmax(outProb) + 1;
    }

    private int predictOnGpu(float[][] x) {
        if (mRs == null) {
            return 0;
        }

//        float[][] a = {{1, 2, 3}, {4, 5, 6}};
//        float[][] b = {{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}};
//        float[][] c1 = {{380, 440, 500, 560}, {830, 980, 1130, 1280}};
        int timeSteps = x.length; // dimY
        int inDim = x[0].length; // dimX
        int outDim = w_out[0].length;
        float[] convertedX = alter2Dto1D(x);
        float[] convertedWIn = alter2Dto1D(w_in);
        float[] convertedWOut = alter2Dto1D(w_out);

        ScriptC_main scriptC_main = new ScriptC_main(mRs);
        scriptC_main.set_timeSteps(timeSteps);
        scriptC_main.set_inputDims(inDim);
        scriptC_main.set_outputDims(outDim);
        scriptC_main.set_hiddenUnites(hidden_units);

        // initialize input allocation
        Type inRawType = Type.createXY(mRs, Element.F32(mRs), inDim, timeSteps);
        Allocation inputRawAlloc = Allocation.createTyped(mRs, inRawType);
        inputRawAlloc.copyFrom(convertedX);
        scriptC_main.set_inputRaw(inputRawAlloc);

        // initialize model parameters allocation
        Type wInType = Type.createXY(mRs, Element.F32(mRs), inDim, hidden_units);
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

        Type inputsType = Type.createXY(mRs, Element.F32(mRs), hidden_units, timeSteps);
        Allocation inputsAlloc = Allocation.createTyped(mRs, inputsType);
        float[] inputs = new float[timeSteps * hidden_units];
//        scriptC_main.set_matAB(inputsAlloc);
//        scriptC_main.set_matA(inputRawAlloc);
//        scriptC_main.set_matB(wInAlloc);
//        scriptC_main.set_sameDim(inDim);
        scriptC_main.forEach_input_transform(inputsAlloc);

        inputsAlloc.copyTo(inputs);
        System.out.println("inputs: " + Arrays.toString(inputs));

        // initialize output label allocation
        Allocation labelProbAlloc = Allocation.createSized(mRs, Element.F32(mRs), outDim);
        scriptC_main.set_label_prob(labelProbAlloc);
        float[] labelProb = new float[outDim];

        // copy result back
        labelProbAlloc.copyTo(labelProb);
        int label = DataUtil.argmax(labelProb) + 1;
        System.out.println(label);
        return label;
    }

    public void setRs(RenderScript rs) {
        mRs = rs;
    }
}
