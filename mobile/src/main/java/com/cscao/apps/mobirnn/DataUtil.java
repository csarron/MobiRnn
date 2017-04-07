package com.cscao.apps.mobirnn;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * Data utility class
 */

public class DataUtil {
    public static double[] parseBias(String fileName) throws IOException {
        List<String> lines = FileUtils.readLines(new File(fileName));
        int size = lines.size();
        double[] b = new double[size];
        for (int i = 0; i < size; i++) {
            b[i] = Double.valueOf(lines.get(i));
        }

        return b;
    }

    public static double[][] parseWeight(String fileName) throws IOException {
        List<String> lines = FileUtils.readLines(new File(fileName));
        int size = lines.size();
        double[][] weights = new double[size][];
        for (int i = 0; i < size; i++) {
            String line = lines.get(i).trim().replaceAll("( )+", ",");
//            System.out.println("line:" + line);
            String[] ws = line.split(",");
            weights[i] = new double[ws.length];
            for (int j = 0; j < ws.length; j++) {
                weights[i][j] = Double.valueOf(ws[j]);
            }
        }
        return weights;
    }

    public static double[][][] parseInputData(String folder) throws IOException {
        String[] inputFileNames = new File(folder).list();
        int input_dim = inputFileNames.length;
        double[][][] data = new double[input_dim][][];

        for (int i = 0; i < input_dim; i++) {
            String filePath = folder + File.separator + inputFileNames[i];
            // System.out.println("filePath: " + filePath);
            data[i] = parseWeight(filePath);
        }

        int samples = data[0].length;
        int steps = data[0][0].length;
        // System.out.println("samples: " + samples + "steps:" + steps);
        double[][][] output = new double[samples][steps][input_dim];
        for (int i = 0; i < input_dim; i++) {
            for (int j = 0; j < samples; j++) {
                for (int k = 0; k < steps; k++) {
                    output[j][k][i] = data[i][j][k];
                    // System.out.println(data[i][j][k]);
                }
            }
        }
        return output;
    }

    public static int[] parseLabel(String fileName) throws IOException {
        List<String> lines = FileUtils.readLines(new File(fileName));

        int size = lines.size();
        int[] label = new int[size];
        for (int i = 0; i < size; i++) {
            label[i] = Integer.valueOf(lines.get(i));
        }

        return label;
    }

    public static double sigmod(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static int argmax(double[] x) {
        int max = 0;
        double a = 0.0f;
        for (int i = 0; i < x.length; i++) {
            if (x[i] > a) {
                a = x[i];
                max = i;
            }
        }
        return max;
    }

    public static double[][] relu(double[][] x) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] = Math.max(x[i][j], 0);
            }
        }

        return x;
    }

}
