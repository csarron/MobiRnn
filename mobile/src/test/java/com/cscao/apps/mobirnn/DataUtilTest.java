package com.cscao.apps.mobirnn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.cscao.apps.mobirnn.model.DataUtil;

import org.junit.Test;

import java.util.Arrays;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * Test weights parsing
 */
public class DataUtilTest {

    @Test
    public void parseBias() throws Exception {
        double[] biases =
                {6.430978178977966309e-01, 1.422856748104095459e-01,
                        1.947010159492492676e+00, -1.941539049148559570e-01,
                        1.498547315597534180e+00, 3.788197934627532959e-01,
                        7.266948223114013672e-01, -1.020129695534706116e-01,
                        5.147905349731445312e-01, 9.793586134910583496e-01};
        String testFilePath = getClass().getClassLoader().getResource("b_in.csv").getFile();

        assertTrue(Arrays.equals(DataUtil.parseBias(testFilePath), biases));

    }

    @Test
    public void parseWeight() throws Exception {

        double[][] weights =
                {{1.302366554737091064e-01, 8.976213634014129639e-02, -3.018706291913986206e-02},
                        {-1.279512979090213776e-02, -9.628687798976898193e-02,
                                1.115116998553276062e-01},
                        {-1.271081268787384033e-01, -7.517708837985992432e-02,
                                7.937213778495788574e-02},
                        {3.289928138256072998e-01, -6.845261901617050171e-02,
                                9.846886992454528809e-01}};

        String testFilePath = getClass().getClassLoader().getResource("w_in.csv").getFile();

        assertTrue(Arrays.deepEquals(DataUtil.parseWeight(testFilePath), weights));
    }

    @Test
    public void parseInputData() throws Exception {
        double[][][] data = {{{1.1653150e-002, -2.9399040e-002, 4.3746370e-001, 5.3134920e-001},
                {1.3109090e-002, -3.9728670e-002, 4.6826410e-001, 7.2106850e-001},
                {1.1268850e-002, -5.2405860e-002, 4.9825740e-001, 5.2032840e-001}},
                {{2.7830730e-002, -5.2106230e-002, 4.7939570e-001, 3.7262520e-001},
                        {2.3183500e-003, -4.5470360e-002, 3.8989350e-001, 4.1454140e-001},
                        {-1.8965500e-002, -3.7763610e-002, 3.0665100e-001, 3.3332790e-001}}};
        String testFilePath = getClass().getClassLoader().getResource("sensor_data").getFile();
        assertTrue(Arrays.deepEquals(DataUtil.parseInputData(testFilePath), data));
    }

    @Test
    public void parseLabel() throws Exception {
        int[] labels = {5, 4, 3, 2};
        String testFilePath = getClass().getClassLoader().getResource("y_test.txt").getFile();
        assertTrue(Arrays.equals(DataUtil.parseLabel(testFilePath), labels));
    }


    @Test
    public void argmax() throws Exception {
        double[] a = {1.2, 3.4, 0.01, 3.22};
        assertEquals(DataUtil.argmax(a), 1);

        double[] b = {1.2, 3.4, 10.01, 3.22};
        assertEquals(DataUtil.argmax(b), 2);

        double[] c = {11.2, 3.4, 0.01, 3.22};
        assertEquals(DataUtil.argmax(c), 0);
    }
}