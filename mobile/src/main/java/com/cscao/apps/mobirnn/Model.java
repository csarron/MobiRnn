package com.cscao.apps.mobirnn;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * LSTM Model class
 */

public class Model {

    public static float[] parseBias(String fileName) {
        List<String> lines;
        try {
             lines = FileUtils.readLines(new File(fileName));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        int size = lines.size();
        float[] b = new float[size];
        for (int i = 0; i < size; i++) {
            b[i] = Float.valueOf(lines.get(i));
        }

        return b;
    }

    public static float[][] parseWeight(String fileName) {
        List<String> lines;
        try {
            lines = FileUtils.readLines(new File(fileName));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        int size = lines.size();
        float[][] weights = new float[size][];
        for (int i = 0; i < size; i++) {
            String[] ws = lines.get(i).split(", ");
            weights[i] = new float[ws.length];
            for (int j = 0; j < ws.length; j++) {
                weights[i][j] = Float.valueOf(ws[j]);
            }
        }
        return weights;
    }
}
