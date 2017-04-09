package com.cscao.apps.mobirnn;

import static com.cscao.apps.mobirnn.model.DataUtil.parseInputData;
import static com.cscao.apps.mobirnn.model.DataUtil.parseLabel;

import static org.junit.Assert.assertTrue;

import com.cscao.apps.mobirnn.model.Model;

import org.junit.Test;

import java.io.File;
import java.util.Arrays;

/**
 * Created by qqcao on 4/6/17.
 *
 * Model tests
 */
public class ModelTest {


    @Test
    public void predict() throws Exception {
        String modelFilePath = getClass().getClassLoader().getResource("lstm_har-data").getFile();
        String inputFilePath =
                modelFilePath + File.separator + "test_data" + File.separator + "sensor";
        Model lstmModel = new Model(modelFilePath);

        float[][][] inputs = parseInputData(inputFilePath);

        int size = inputs.length;
        int[] predictedLabels = new int[size];
        long begin = System.currentTimeMillis();
        for (int i = 0; i < size; i++) {
            predictedLabels[i] = lstmModel.predict(inputs[i]);
//            System.out.println(predictedLabels[i]);
        }
        long end = System.currentTimeMillis();
        System.out.println("time spent: " + (end - begin) / 1000.0);
        String result = Arrays.toString(predictedLabels).
                replaceAll("\\[", "").replaceAll("\\]", "").replaceAll(", ", "\n");

//        File labelsFile = new File(modelFilePath + "jLables.txt");
//        System.out.println(labelsFile.getAbsolutePath());
//        FileUtils.writeStringToFile(labelsFile, result);

        String labelFilePath = getClass().getClassLoader().getResource("labels_np.log").getFile();

        int[] labels = parseLabel(labelFilePath);

        assertTrue(Arrays.equals(predictedLabels, labels));

    }

}