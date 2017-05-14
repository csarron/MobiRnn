package com.cscao.apps.mobirnn.helper;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Created by qqcao on 4/7/17.
 *
 * Android Util
 */

public class Util {


    public static String getTimestampString() {
        SimpleDateFormat df = new SimpleDateFormat("mm:ss.SSS", Locale.US);
        return df.format(new Date());
    }

}
