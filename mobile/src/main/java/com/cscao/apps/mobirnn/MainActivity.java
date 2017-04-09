package com.cscao.apps.mobirnn;

import static com.cscao.apps.mobirnn.helper.Util.getInputData;
import static com.cscao.apps.mobirnn.helper.Util.getLabels;
import static com.cscao.apps.mobirnn.helper.Util.getTimestampString;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.NumberPicker;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.cscao.apps.mobirnn.helper.Util;
import com.cscao.apps.mobirnn.model.Model;
import com.orhanobut.logger.Logger;

import java.io.IOException;
import java.util.Locale;

public class MainActivity extends Activity implements NumberPicker.OnValueChangeListener {

    //    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }
    public final int PERMISSIONS_REQUEST_CODE = 7;
    private ToggleButton controlToggle;
    private TextView mStatusTextView;
    private TextView mResultTextView;
    private ProgressBar mResultProgress;

    private Task mTask = new Task();
    private boolean mIsCpuMode = true;
    private int mSampleSize;
    final String[] mSampleSizes = {"10", "50", "100", "200", "500", "1000"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        controlToggle = (ToggleButton) findViewById(R.id.toggle_control);

        mStatusTextView = (TextView) findViewById(R.id.tv_status);
        mStatusTextView.setMovementMethod(new ScrollingMovementMethod());
        mResultTextView = (TextView) findViewById(R.id.tv_result);
        mResultProgress = (ProgressBar) findViewById(R.id.progress);
        mResultProgress.setMax(100);

        NumberPicker picker = (NumberPicker) findViewById(R.id.number_picker);
        picker.setDescendantFocusability(NumberPicker.FOCUS_BLOCK_DESCENDANTS);
        picker.setOnValueChangedListener(this);
        picker.setDisplayedValues(mSampleSizes);
        picker.setMinValue(0);
        picker.setMaxValue(mSampleSizes.length - 1);
        picker.setWrapSelectorWheel(true);
        picker.setValue(2);
        mSampleSize = 100;
        Logger.i("Sample size initial value: %s", mSampleSize);

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
                Logger.d("Permission granted");
            } else {
                Toast.makeText(this, "Please grant write permission", Toast.LENGTH_LONG).show();
            }
        }
    }


    public void onRadioButtonClicked(View view) {
        switch (view.getId()) {
            case R.id.radio_cpu:
                mIsCpuMode = true;
                Logger.d("selected cpu mode");
                break;
            case R.id.radio_gpu:
                mIsCpuMode = false;
                Logger.d("selected gpu mode");
                break;
            default:
                mIsCpuMode = true;
        }
    }

    public void controlRun(View view) {
        if (controlToggle.isChecked()) {
            mResultTextView.setText("");
            mTask = new Task();
            mTask.mIsCpuMode = mIsCpuMode;
            mTask.mSampleSize = mSampleSize;
            Logger.i("running task");
            mTask.execute(Util.getDataPath());
            Toast.makeText(this, R.string.run_model, Toast.LENGTH_SHORT).show();
        } else {
            mTask.cancel(true);
            setTaskCancellationInfo("User stopped task, ");
        }
    }

    private void setTaskCancellationInfo(String info) {
        Logger.i("task cancelled");
        Toast.makeText(this, R.string.task_cancelled, Toast.LENGTH_SHORT).show();
        String status = String.format(Locale.US, "%s: %s, task cancelled\n",
                getTimestampString(), info);
        mStatusTextView.append(status);
        mResultProgress.setProgress(0);
        mResultTextView.setText("");
    }

    @Override
    public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
        Logger.i("Sample size changed from %s to %s", mSampleSizes[oldVal], mSampleSizes[newVal]);
        mSampleSize = Integer.parseInt(mSampleSizes[newVal]);
    }


    private class Task extends AsyncTask<String, String, Pair<Double, Double>> {

        private boolean mIsCpuMode;
        private int mSampleSize;

        @Override
        protected Pair<Double, Double> doInBackground(String... params) {
            String dataRootPath = params[0];
            Model lstmModel = null;
            Log.d("run", "begin model loading");
            try {
                lstmModel = new Model(dataRootPath, mIsCpuMode);
                publishProgress("0", "model loaded");
            } catch (IOException e) {
                Logger.e("model cannot be created");
                e.printStackTrace();
                publishProgress("-1", "model cannot be created");
                this.cancel(true);
            }
            Log.d("run", "model created");

            double[][][] inputs = new double[0][][];
            try {
                publishProgress("0", "begin parsing input data...");
                inputs = getInputData(dataRootPath, mSampleSize);
                publishProgress("0", "input data loaded");
            } catch (IOException e) {
                Logger.e("input data cannot be parsed");
                publishProgress("-1", "input data cannot be parsed");
                e.printStackTrace();
                this.cancel(true);
            }

            int[] labels = new int[0];
            try {
                labels = getLabels(dataRootPath, mSampleSize);
                publishProgress("0", "labels loaded");
            } catch (IOException e) {
                Logger.e("label data cannot be parsed");
                publishProgress("-1", "label data cannot be parsed");
                e.printStackTrace();
                this.cancel(true);
            }

            int[] predictedLabels = new int[mSampleSize];

            long beginTime = System.currentTimeMillis();
            int correct = 0;
            for (int i = 0; i < mSampleSize; i++) {
                if (this.isCancelled()) {
                    break;
                }
                assert lstmModel != null;
                predictedLabels[i] = lstmModel.predict(inputs[i]);
                boolean isCorrect = (predictedLabels[i] == labels[i]);
                if (isCorrect) {
                    correct++;
                }
                String progress = String.format(Locale.US,
                        "case: %d, label: %d, correct: %s", i, predictedLabels[i], isCorrect);
                Logger.d(progress);
                publishProgress("" + (i+1), progress);

            }
            long endTime = System.currentTimeMillis();

            double accuracy = correct * 100.0 / mSampleSize;
            double time = (endTime - beginTime) / 1000.0;
            return Pair.create(accuracy, time);
        }

        @Override
        protected void onPreExecute() {
            String mode = String.format(Locale.US,
                    "running model on %s\n", mIsCpuMode ? "CPU" : "GPU");
            mStatusTextView.append(mode);

            String info = String.format(Locale.US,
                    "%s: processing model and input data...\n", getTimestampString());
            mStatusTextView.append(info);
        }

        @Override
        protected void onPostExecute(Pair<Double, Double> pair) {
            double accuracy = pair.first;
            double time = pair.second;
            String show = String.format(Locale.US,
                    "Accuracy is: %s  %%. Time spent: %s s", accuracy, time);
            mResultTextView.setText(show);

            String status = String.format(Locale.US, "%s: task finished\n", getTimestampString());
            mStatusTextView.append(status);

            controlToggle.setChecked(false);
        }

        @Override
        protected void onProgressUpdate(String... values) {
            int i = Integer.valueOf(values[0]);
            String status = String.format(Locale.US, "%s: %s\n", getTimestampString(), values[1]);

            if (i == -1) {
                setTaskCancellationInfo(status);
                return;
            }

            int progress = (int) ((i / (float) mSampleSize) * 100);
            mResultProgress.setProgress(progress);
            mStatusTextView.append(status);
        }
    }

//    public native String stringFromJNI();
}
