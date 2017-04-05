package com.cscao.apps.mobirnn;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;

/**
 * Created by qqcao on 4/5/17Wednesday.
 *
 * Test weights parsing
 */
public class ModelTest {

    @Test
    public void parseBias() throws Exception {
        float[] biases =
                {6.430978178977966309e-01f, 1.422856748104095459e-01f,
                        1.947010159492492676e+00f, -1.941539049148559570e-01f,
                        1.498547315597534180e+00f, 3.788197934627532959e-01f,
                        7.266948223114013672e-01f, -1.020129695534706116e-01f,
                        5.147905349731445312e-01f, 9.793586134910583496e-01f};
        String testFilePath = getClass().getClassLoader().getResource("b_in.csv").getFile();

        assertTrue(Arrays.equals(Model.parseBias(testFilePath), biases));

    }

    @Test
    public void parseWeight() throws Exception {

        float[][] weights =
                {{1.302366554737091064e-01f, 8.976213634014129639e-02f, -3.018706291913986206e-02f},
                        {-1.279512979090213776e-02f, -9.628687798976898193e-02f,
                                1.115116998553276062e-01f},
                        {-1.271081268787384033e-01f, -7.517708837985992432e-02f,
                                7.937213778495788574e-02f},
                        {3.289928138256072998e-01f, -6.845261901617050171e-02f,
                                9.846886992454528809e-01f}};

        String testFilePath = getClass().getClassLoader().getResource("w_in.csv").getFile();

        assertTrue(Arrays.deepEquals(Model.parseWeight(testFilePath), weights));
    }

}