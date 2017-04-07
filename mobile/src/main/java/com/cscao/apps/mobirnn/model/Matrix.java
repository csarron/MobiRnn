package com.cscao.apps.mobirnn.model;

/**
 * Created by qqcao on 4/6/17
 *
 * collection of static methods for matrix operations
 */
public class Matrix {

    // return B = A^T
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                b[j][i] = a[i][j];
            }
        }
        return b;
    }

    // return c = a + b
    public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    public static double[][] addVec(double[][] m, double[] v) {
        double[][] r = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            int rowSize = m[i].length;
            r[i] = new double[rowSize];
            for (int j = 0; j < rowSize; j++) {
                r[i][j] = m[i][j] + v[j];
            }
        }
        return r;
    }

    public static double[] vecAddVec(double[] a, double[] b) {
        int m = a.length;
        double[] c = new double[m];
        for (int i = 0; i < m; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    // return c = a * b
    public static double[][] multiply(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("dimension of a and b must be the same");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n1; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }

    // vector-matrix multiplication (y = x^T A)
    public static double[] vecMulMat(double[] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                y[j] += a[i][j] * x[i];
            }
        }
        return y;
    }

    public static double[] concat(double[] a, double[] b) {
        int aSize = a.length;
        int bSize = b.length;
        if (aSize != bSize) {
            throw new RuntimeException(
                    "dimension of a:" + aSize + " and b:" + bSize + " must be the same");
        }
        double[] concat = new double[aSize + bSize];
        System.arraycopy(a, 0, concat, 0, aSize);
        System.arraycopy(b, 0, concat, aSize, bSize);

        return concat;
    }

    public static double[] split(double[] a, int splitSize, int resultIndex) {
        int partitionSize = a.length / splitSize;
        if (resultIndex >= splitSize) {
            throw new RuntimeException("resultIndex:" + resultIndex
                    + " must be smaller than splitSize:" + splitSize);
        }
        double[] r = new double[partitionSize];
        System.arraycopy(a, resultIndex * partitionSize, r, 0, partitionSize);
        return r;
    }
}
