package com.cscao.apps.mobirnn.helper;

import android.os.Environment;

import java.io.File;

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
}
