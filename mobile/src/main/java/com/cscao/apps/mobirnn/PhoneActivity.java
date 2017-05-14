package com.cscao.apps.mobirnn;

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

import com.cscao.apps.mobirnn.model.Model;
import com.cscao.apps.mobirnn.model.ModelMode;
import com.orhanobut.logger.Logger;

import java.util.Locale;
import java.util.Random;

public class PhoneActivity extends Activity implements NumberPicker.OnValueChangeListener {

    //    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }
    public final int PERMISSIONS_REQUEST_CODE = 7;
    private ToggleButton controlToggle;
    private TextView mStatusTextView;
    private ProgressBar mResultProgress;

    private Task mTask;
    private int mSeed;
    private ModelMode mRunMode = ModelMode.TensorFlow;
    private int mSampleSize;
    final String[] mSampleSizes = {"1", "10", "50", "100", "200", "500"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        controlToggle = (ToggleButton) findViewById(R.id.toggle_control);

        mStatusTextView = (TextView) findViewById(R.id.tv_status);
        mStatusTextView.setMovementMethod(new ScrollingMovementMethod());
        mResultProgress = (ProgressBar) findViewById(R.id.progress);
        mResultProgress.setMax(100);

        NumberPicker picker = (NumberPicker) findViewById(R.id.number_picker);
        picker.setDescendantFocusability(NumberPicker.FOCUS_BLOCK_DESCENDANTS);
        picker.setOnValueChangedListener(this);
        picker.setDisplayedValues(mSampleSizes);
        picker.setMinValue(0);
        picker.setMaxValue(mSampleSizes.length - 1);
        picker.setWrapSelectorWheel(true);
        picker.setValue(0);
        mSampleSize = 1;
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
                mRunMode = ModelMode.CPU;
                Logger.d("selected cpu mode");
                break;
            case R.id.radio_gpu:
                mRunMode = ModelMode.GPU;
                Logger.d("selected gpu mode");
                break;
            case R.id.radio_tf:
                mRunMode = ModelMode.TensorFlow;
                Logger.d("selected tensorflow mode");
                break;
            default:
                mRunMode = ModelMode.TensorFlow;
        }
    }

    public void controlRun(View view) {
        if (controlToggle.isChecked()) {
            mStatusTextView.setText("");
            mTask = new Task();
            mTask.mMode = mRunMode;
            Logger.i("running task");
            mTask.execute(mSampleSize, mSeed);
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
        controlToggle.setChecked(false);
    }

    @Override
    public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
//        Logger.i("Sample size changed from %s to %s", mSampleSizes[oldVal], mSampleSizes[newVal]);
        mSampleSize = Integer.parseInt(mSampleSizes[newVal]);
    }

    public void changeSeed(View view) {
        mSeed = (int) System.currentTimeMillis();
        Toast.makeText(this, R.string.seed_changed, Toast.LENGTH_SHORT).show();
    }


    private class Task extends AsyncTask<Integer, String, Pair<Float, Float>> {
        private ModelMode mMode;

        private int[] getSampledLabels(int[] labels, int[] indices) {
            int size = indices.length;
            int[] sampledLabels = new int[size];
            for (int i = 0; i < size; i++) {
                sampledLabels[i] = labels[indices[i]];
            }
            return sampledLabels;
        }

        private float[][] getSampledInputs(float[][] inputs, int[] indices) {
            int size = indices.length;
            float[][] sampledInputs = new float[size][];
            for (int i = 0; i < size; i++) {
                sampledInputs[i] = inputs[indices[i]];
            }
            return sampledInputs;
        }

        @NonNull
        private int[] getIndices(int high, int size, int seed) {
            int[] indices = new int[size];
            Random r = new Random(seed);
            for (int i = 0; i < indices.length; i++) {
                indices[i] = r.nextInt(high);
            }

//            Logger.d("indices: " + Arrays.toString(indices));
            return indices;
        }

        @Override
        protected Pair<Float, Float> doInBackground(Integer... params) {
            int sampleSize = params[0];
            int seed = params[1];

            Log.d("run", "begin model loading");
            Model model = new Model(getApplicationContext(), mMode, 2, 32);

            publishProgress("0", "model loaded");
            Log.d("run", "model created");

            publishProgress("0", "loading input data...");
            float[][] inputs = model.loadInputs();
            int[] indices = getIndices(inputs.length, sampleSize, seed);

            inputs = getSampledInputs(inputs, indices);
            publishProgress("0", "input data loaded");

            int[] labels = model.loadLabels();
            labels = getSampledLabels(labels, indices);
            publishProgress("0", "labels loaded");

            int[] predictedLabels = new int[sampleSize];

            long beginTime = System.currentTimeMillis();
            int correct = 0;
            float accuracy = 0;
            for (int i = 0; i < sampleSize; i++) {
                if (this.isCancelled()) {
                    break;
                }
                predictedLabels[i] = model.predict(inputs[i]);
                boolean isCorrect = (predictedLabels[i] == labels[i]);
                if (isCorrect) {
                    correct++;
                    accuracy = (float) (correct * 100.0 / (i + 1));
                }
                String progress = String.format(Locale.US,
                        "case:%03d,output:%d,label:%d,%s",
                        indices[i], predictedLabels[i], labels[i], isCorrect ? "right" : "wrong");
                Logger.d(progress);
                publishProgress("" + (i + 1), progress);

            }
            long endTime = System.currentTimeMillis();

            float time = (float) ((endTime - beginTime) / 1000.0);
            return Pair.create(accuracy, time);
        }

        @Override
        protected void onPreExecute() {
            String mode = String.format(Locale.US,
                    "running model in %s mode\n", mMode.toString());
            mStatusTextView.append(mode);

            String info = String.format(Locale.US,
                    "%s: loading model...\n", getTimestampString());
            mStatusTextView.append(info);
            mResultProgress.setProgress(0);
        }

        @Override
        protected void onCancelled(Pair<Float, Float> pair) {
            updateUIUponTaskEnding(pair);
        }

        @Override
        protected void onPostExecute(Pair<Float, Float> pair) {
            updateUIUponTaskEnding(pair);
        }

        private void updateUIUponTaskEnding(Pair<Float, Float> pair) {
            if (pair != null) {
                float accuracy = pair.first;
                float time = pair.second;
                String show = String.format(Locale.US,
                        "Accuracy is: %s  %%. Time spent: %s s", accuracy, time);
                String status = String.format(Locale.US, "%s: task finished\n",
                        getTimestampString());
                mStatusTextView.append(status);
                mStatusTextView.append(show);

            }

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


    @Override
    protected void onSaveInstanceState(Bundle outState) {

        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
    }

}
