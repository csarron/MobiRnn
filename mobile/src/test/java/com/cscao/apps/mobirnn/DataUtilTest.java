package com.cscao.apps.mobirnn;

import static com.cscao.apps.mobirnn.helper.DataUtil.sigmoid;
import static com.cscao.apps.mobirnn.helper.DataUtil.tanh;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.cscao.apps.mobirnn.helper.DataUtil;

import org.junit.Test;

import java.util.Arrays;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * Test weights parsing
 */
public class DataUtilTest {
    @Test
    public void alter3Dto1D() throws Exception {
        float[][][] a = {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
                {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}};
        float[] b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        assertTrue(Arrays.equals(DataUtil.alter3Dto1D(a), b));
    }

    @Test
    public void alter2Dto1D() throws Exception {
        float[][] a = {{1, 2}, {3, 4}, {5, 6}};
        float[] b = {1, 2, 3, 4, 5, 6};
        assertTrue(Arrays.equals(DataUtil.alter2Dto1D(a), b));
    }

    @Test
    public void calc() throws Exception{
        float cVal = 0f;
        float fVal = 2.104409f;
        float iVal = 0.378655f;
        float jVal = -1.484427f;
        float oVal = 3.171872f;
        float newC = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanh(jVal);
        float newH = tanh(newC) * sigmoid(oVal);
        System.out.println("newC: " + newC);
        System.out.println("newH: " + newH);

        System.out.println("tanh 3: " + tanh(3));
    }

    @Test
    public void argmax() throws Exception {
        float[] a = {1.2f, 3.4f, 0.01f, 3.22f};
        assertEquals(DataUtil.argmax(a), 1);

        float[] b = {1.2f, 3.4f, 10.01f, 3.22f};
        assertEquals(DataUtil.argmax(b), 2);

        float[] c = {11.2f, 3.4f, 0.01f, 3.22f};
        assertEquals(DataUtil.argmax(c), 0);
    }
}