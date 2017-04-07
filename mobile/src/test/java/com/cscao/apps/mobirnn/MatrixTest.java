package com.cscao.apps.mobirnn;

import static org.junit.Assert.assertTrue;

import com.cscao.apps.mobirnn.model.Matrix;

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
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[][] b = {{1, 4}, {2, 5}, {3, 6}};
        assertTrue(Arrays.deepEquals(Matrix.transpose(a), b));
    }

    @Test
    public void add() throws Exception {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[][] b = {{7, 8, 9}, {10, 11, 12}};
        double[][] c = {{8, 10, 12}, {14, 16, 18}};
        assertTrue(Arrays.deepEquals(Matrix.add(a, b), c));
    }

    @Test
    public void addVec() throws Exception {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[] b = {7, 8, 9};
        double[][] c = {{8, 10, 12}, {11, 13, 15}};
        assertTrue(Arrays.deepEquals(Matrix.addVec(a, b), c));
    }

    @Test
    public void vecAddVec() throws Exception {
        double[] a = {1, 2, 3, 4, 5, 6};
        double[] b = {7, 8, 9, 10, 11, 12};
        double[] c = {8, 10, 12, 14, 16, 18};
        assertTrue(Arrays.equals(Matrix.vecAddVec(a, b), c));
    }

    @Test
    public void multiply() throws Exception {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[][] b = {{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}};
        double[][] c = {{380, 440, 500, 560}, {830, 980, 1130, 1280}};
        assertTrue(Arrays.deepEquals(Matrix.multiply(a, b), c));
    }

    @Test
    public void vecMulMat() throws Exception {
        double[] a = {1, 2, 3};
        double[][] b = {{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}};
        double[] c = {380, 440, 500, 560};
        assertTrue(Arrays.equals(Matrix.vecMulMat(a, b), c));
    }

    @Test
    public void concat() throws Exception {
        double[] d = {1, 2, 3};
        double[] e = {4, 5, 6};
        double[] f = {1, 2, 3, 4, 5, 6};
        assertTrue(Arrays.equals(Matrix.concat(d, e), f));
    }

    @Test
    public void split() throws Exception {
        double[] a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        double[] a0 = {1, 2, 3, 4};
        double[] a1 = {5, 6, 7, 8};
        double[] a2 = {9, 10, 11, 12};
        assertTrue(Arrays.equals(Matrix.split(a, 3, 0), a0));
        assertTrue(Arrays.equals(Matrix.split(a, 3, 1), a1));
        assertTrue(Arrays.equals(Matrix.split(a, 3, 2), a2));
    }
}