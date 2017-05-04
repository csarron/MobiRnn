package com.cscao.apps.mobirnn.model;

/**
 * Created by qqcao on 4/6/17
 *
 * collection of static methods for matrix operations
 */
public class Matrix {

    // return B = A^T
    public static float[][] transpose(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] b = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                b[j][i] = a[i][j];
            }
        }
        return b;
    }

    // return c = a + b
    public static float[][] add(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        float[][] c = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    public static float[][] addVec(float[][] m, float[] v) {
        float[][] r = new float[m.length][];
        for (int i = 0; i < m.length; i++) {
            int rowSize = m[i].length;
            r[i] = new float[rowSize];
            for (int j = 0; j < rowSize; j++) {
                r[i][j] = m[i][j] + v[j];
            }
        }
        return r;
    }

    public static float[] vecAddVec(float[] a, float[] b) {
        int m = a.length;
        float[] c = new float[m];
        for (int i = 0; i < m; i++) {
            c[i] = a[i] + b[i];
        }
        return c;
    }

    // return c = a * b
    public static float[][] multiply(float[][] a, float[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("dimension of a and b must be the same");
        float[][] c = new float[m1][n2];
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
    public static float[] vecMulMat(float[] x, float[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        float[] y = new float[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                y[j] += a[i][j] * x[i];
            }
        }
        return y;
    }

    public static float[] concat(float[] a, float[] b) {
        int aSize = a.length;
        int bSize = b.length;
        if (aSize != bSize) {
            throw new RuntimeException(
                    "dimension of a:" + aSize + " and b:" + bSize + " must be the same");
        }
        float[] concat = new float[aSize + bSize];
        System.arraycopy(a, 0, concat, 0, aSize);
        System.arraycopy(b, 0, concat, aSize, bSize);

        return concat;
    }

    public static float[] split(float[] a, int splitSize, int resultIndex) {
        int partitionSize = a.length / splitSize;
        if (resultIndex >= splitSize) {
            throw new RuntimeException("resultIndex:" + resultIndex
                    + " must be smaller than splitSize:" + splitSize);
        }
        float[] r = new float[partitionSize];
        System.arraycopy(a, resultIndex * partitionSize, r, 0, partitionSize);
        return r;
    }
}
