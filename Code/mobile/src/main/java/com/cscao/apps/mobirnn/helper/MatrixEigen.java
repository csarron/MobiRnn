package com.cscao.apps.mobirnn.helper;

/**
 * Created by qqcao on 4/6/17
 *
 * collection of static methods for matrix operations
 */
public class MatrixEigen {

    public static native float[][] addVec(float[][] m, float[] v);

    public static native float[] vecAddVec(float[] a, float[] b);

    // return c = a * b
    public static native float[][] multiply(float[][] a, float[][] b);

    // vector-matrix multiplication (y = x^T A)
    public static native float[] vecMulMat(float[] x, float[][] a);

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
