package com.cscao.apps.mobirnn.helper;

import static com.cscao.apps.mobirnn.model.DataUtil.parseInputData;
import static com.cscao.apps.mobirnn.model.DataUtil.parseLabel;

import android.os.Environment;

import com.orhanobut.logger.Logger;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Created by qqcao on 4/7/17.
 *
 * Android Util
 */

public class Util {
    public static final String DATA_URL = "https://github.com/csarron/lstm_har/archive/data.zip";
    public static final String folder = "lstm_har-data";

    public static String getDataPath() {
        File sdcard = Environment.getExternalStorageDirectory();
        File dir = new File(sdcard.getPath() + File.separator + folder);
        return dir.getAbsolutePath();
    }

    public static String getTimestampString() {
        SimpleDateFormat df = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);
        return df.format(new Date());
    }

    public static double[][][] getInputData(String folder, int sampleSize) throws IOException {
        Logger.i("begin parsing input data");
        String inputFilePath = folder + File.separator + "test_data" + File.separator + "sensor";
        double[][][] inputs = parseInputData(inputFilePath);
        Logger.i("end parsing input data");

        return inputs;
    }

    public static int[] getLabels(String folder, int sampleSize) throws IOException {
        String labelPath = folder + File.separator + "test_data" + File.separator + "y_test.txt";
        return parseLabel(labelPath);
    }
}
