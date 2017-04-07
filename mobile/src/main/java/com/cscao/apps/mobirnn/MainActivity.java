package com.cscao.apps.mobirnn;

import static com.cscao.apps.mobirnn.model.DataUtil.parseInputData;
import static com.cscao.apps.mobirnn.model.DataUtil.parseLabel;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.cscao.apps.mobirnn.helper.Util;
import com.cscao.apps.mobirnn.model.Model;

import java.io.File;
import java.io.IOException;

public class MainActivity extends Activity {

    //    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }
    public final int PERMISSIONS_REQUEST_CODE = 7;

    private TextView mCpuTimeTextView;
    private TextView mGpuTimeTextView;
    private Task mTask = new Task();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mCpuTimeTextView = (TextView) findViewById(R.id.tv_time_cpu);
        mGpuTimeTextView = (TextView) findViewById(R.id.tv_time_gpu);
        // Example of a call to a native method
//        TextView tv = (TextView) findViewById(R.id.sample_text);
//        tv.setText(stringFromJNI());
        checkPermissions();

    }

    private void checkPermissions() {
        boolean writeExternalStoragePermissionGranted =
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        == PackageManager.PERMISSION_GRANTED;

        if (!writeExternalStoragePermissionGranted) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSIONS_REQUEST_CODE);
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
            @NonNull String permissions[], @NonNull int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            } else {
                Toast.makeText(this, "Please grant write permission", Toast.LENGTH_LONG).show();
            }
        }
    }

    public void runOnCpu(View view) {
        String dataPath = Util.getDataPath();
        mTask.execute(dataPath);
    }

    public void runOnGpu(View view) {
    }


    private class Task extends AsyncTask<String, Integer, Pair<Double, Double>> {

        @Override
        protected Pair<Double, Double> doInBackground(String... params) {
            String modelFilePath = params[0];
            Model lstmModel = null;
            Log.d("run", "begin model loading");
            try {
                lstmModel = new Model(modelFilePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
            Log.d("run", "model created");

            String labelFilePath =
                    modelFilePath + File.separator + "test_data" + File.separator + "y_test.txt";
            String inputFilePath =
                    modelFilePath + File.separator + "test_data" + File.separator + "sensor";

            double[][][] inputs = new double[0][][];
            try {
                inputs = parseInputData(inputFilePath);
                Log.d("run", "input data parsed");
            } catch (IOException e) {
                e.printStackTrace();
            }

            int size = 100;
            int[] predictedLabels = new int[size];
            long beginTime = System.currentTimeMillis();
            for (int i = 0; i < size; i++) {
                assert lstmModel != null;
                predictedLabels[i] = lstmModel.predict(inputs[i]);
                Log.d("run", "predicted case:" + i + ", label: " + predictedLabels[i]);
//            System.out.println(predictedLabels[i]);
            }
            long endTime = System.currentTimeMillis();
            int[] labels = new int[0];
            try {
                labels = parseLabel(labelFilePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
            int correct = 0;
            for (int i = 0; i < size; i++) {
                if (predictedLabels[i] == labels[i]) {
                    correct++;
                }
            }
            double accuracy = correct * 100.0 / size;
            double time = (endTime - beginTime) / 1000.0;
            return Pair.create(accuracy, time);
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected void onPostExecute(Pair<Double, Double> pair) {
            double accuracy = pair.first;
            double time = pair.second;
            String show = "Accuracy is: " + accuracy + "%. " + "Time spent: " + time + " s";

            mCpuTimeTextView.setText(show);
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }
    }

//    public native String stringFromJNI();
}
