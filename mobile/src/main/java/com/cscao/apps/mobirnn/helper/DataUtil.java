package com.cscao.apps.mobirnn.helper;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * Data utility class
 */

public class DataUtil {

    public static float[] alter2Dto1D(float[][] x) {
        int X = x.length;
        int Y = x[0].length;
        float[] converted = new float[X * Y];
        for (int i = 0; i < X; i++) {
            System.arraycopy(x[i], 0, converted, i * Y, Y);
        }
        return converted;
    }

    public static float[] alter3Dto1D(float[][][] x) {
        int X = x.length;
        int Y = x[0].length;
        int Z = x[0][0].length;

        float[] converted = new float[X * Y * Z];
        for (int i = 0; i < X; i++) {
            float[][] layer = x[i];
            for (int j = 0; j < Y; j++) {
                System.arraycopy(layer[j], 0, converted, i * Y * Z + j * Z, Z);
            }
        }
        return converted;
    }

    public static float sigmoid(float x) {
        return 1 / (1 + (float) Math.exp(-x));
    }

    public static float tanh(float x) {
        return (float) Math.tanh(x);
    }

    public static int argmax(float[] x) {
        int max = 0;
        float a = 0.0f;
        for (int i = 0; i < x.length; i++) {
            if (x[i] > a) {
                a = x[i];
                max = i;
            }
        }
        return max;
    }

    public static float[][] relu(float[][] x) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] = Math.max(x[i][j], 0);
            }
        }

        return x;
    }

}
