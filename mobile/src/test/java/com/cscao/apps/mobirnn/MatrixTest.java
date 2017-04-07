package com.cscao.apps.mobirnn;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;

/**
 * Created by qqcao on 4/6/17.
 *
 * Matrix tests
 */
public class MatrixTest {

    @Test
    public void transpose() throws Exception {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{1, 4}, {2, 5}, {3, 6}};
        assertTrue(Arrays.deepEquals(Matrix.transpose(a), b));
    }

    @Test
    public void add() throws Exception {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{7, 8, 9}, {10, 11, 12}};
        float[][] c = {{8, 10, 12}, {14, 16, 18}};
        assertTrue(Arrays.deepEquals(Matrix.add(a, b), c));
    }

    @Test
    public void addVec() throws Exception {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[] b = {7, 8, 9};
        float[][] c = {{8, 10, 12}, {11, 13, 15}};
        assertTrue(Arrays.deepEquals(Matrix.addVec(a, b), c));
    }

    @Test
    public void multiply() throws Exception {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}};
        float[][] c = {{380, 440, 500, 560}, {830, 980, 1130, 1280}};
        assertTrue(Arrays.deepEquals(Matrix.multiply(a, b), c));
    }

    @Test
    public void concat() throws Exception {
        float[][] a = {{1, 2, 3}, {4, 5, 6}};
        float[][] b = {{11, 12, 13}, {14, 15, 16}};
        float[][] c = {{1, 2, 3}, {4, 5, 6}, {11, 12, 13}, {14, 15, 16}};
        assertTrue(Arrays.deepEquals(Matrix.concat(a, b), c));
    }

    @Test
    public void split() throws Exception {
        float[][] a = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
        float[][] a0 = {{1, 2}, {7, 8}};
        float[][] a1 = {{3, 4}, {9, 10}};
        float[][] a2 = {{5, 6}, {11, 12}};
        assertTrue(Arrays.deepEquals(Matrix.split(a, 3, 0), a0));
        assertTrue(Arrays.deepEquals(Matrix.split(a, 3, 1), a1));
        assertTrue(Arrays.deepEquals(Matrix.split(a, 3, 2), a2));

        float[][] a3 = {{1, 2, 3}, {7, 8, 9}};
        float[][] a4 = {{4, 5, 6}, {10, 11, 12}};
        assertTrue(Arrays.deepEquals(Matrix.split(a, 2, 0), a3));
        assertTrue(Arrays.deepEquals(Matrix.split(a, 2, 1), a4));
    }
}