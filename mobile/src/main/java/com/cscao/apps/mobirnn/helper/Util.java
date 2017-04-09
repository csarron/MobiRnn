package com.cscao.apps.mobirnn.helper;

import static com.cscao.apps.mobirnn.model.DataUtil.parseInputData;
import static com.cscao.apps.mobirnn.model.DataUtil.parseLabel;

import android.os.Environment;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.orhanobut.logger.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
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
    // TODO: 4/9/17 if lstm_har-data not existed on sdcard, then download it
    public static final String DATA_URL = "https://github.com/csarron/lstm_har/archive/data.zip";
    public static final String folder = "lstm_har-data";
    private static float[][][] cachedInputs;

    public static String getDataPath() {
        File sdcard = Environment.getExternalStorageDirectory();
        File dir = new File(sdcard.getPath() + File.separator + folder);
        return dir.getAbsolutePath();
    }

    public static String getTimestampString() {
        SimpleDateFormat df = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);
        return df.format(new Date());
    }

    public static float[][][] getInputData(String folder) throws IOException {
        if (cachedInputs != null) {
            return cachedInputs;
        }

        final float[][][] inputs;
        final Kryo kryo = new Kryo();
        kryo.register(float[][][].class);
        final File dataBinFile = new File(getDataPath() + File.separator + "data.bin");
        if (dataBinFile.exists()) {
            Logger.i("begin reading input data bin: %s", dataBinFile.getAbsolutePath());
            Input input = new Input(new FileInputStream(dataBinFile));
            inputs = kryo.readObject(input, float[][][].class);
            input.close();
            Logger.i("begin reading input data bin: %s", dataBinFile.getAbsolutePath());
        } else {
            Logger.i("begin parsing input data");
            String inputFilePath =
                    folder + File.separator + "test_data" + File.separator + "sensor";
            inputs = parseInputData(inputFilePath);
            Logger.i("end parsing input data");
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        Output output = new Output(new FileOutputStream(dataBinFile));
                        kryo.writeObject(output, inputs);
                        output.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
            }).start();
        }

        cachedInputs = inputs;
        return cachedInputs;
    }

    public static int[] getLabels(String folder) throws IOException {
        String labelPath = folder + File.separator + "test_data" + File.separator + "y_test.txt";
        return parseLabel(labelPath);
    }
}
