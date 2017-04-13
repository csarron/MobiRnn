#pragma version(1)
#pragma rs java_package_name(com.cscao.apps.mobirnn)

rs_allocation inputRaw;
uint32_t timeSteps;
uint32_t inputDims;

rs_allocation w_in;
rs_allocation b_in;

float RS_KERNEL input_transform(uint32_t x, uint32_t y){
    float sum = 0.0f;
    for (uint32_t c = 0; c < inputDims; c++) {
        float a = rsGetElementAt_float(inputRaw, c, y);
        float b = rsGetElementAt_float(w_in, x, c);
        sum += a * b;
    }
    float valB = rsGetElementAt_float(b_in, x);
    sum += valB;
    return fmax(sum, 0.0f);// relu
}

rs_allocation weights;
rs_allocation biases;
uint32_t layerSize;
uint32_t hiddenUnites;
rs_allocation inputs;  // timeSteps * hiddenUnits

void input_transform_func(){
    for (uint32_t x = 0; x < hiddenUnites; x++) {
        for (uint32_t y = 0; y < timeSteps; y++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < inputDims; c++) {
                float a = rsGetElementAt_float(inputRaw, c, y);
                float b = rsGetElementAt_float(w_in, x, c);
                sum += a * b;
            }
            float valB = rsGetElementAt_float(b_in, x);
            rsSetElementAt_float(inputs, fmax(sum + valB, 0.0f), x, y);
        }
   }
}

float RS_KERNEL set_zeros(uint32_t x, uint32_t y){
    return 0.0f;
}

float* c;
float* h;

void set_ch_zeros(){
    for (uint32_t x = 0; x < hiddenUnites; x++) {
        h[x] = 0;
        c[x] = 0;
    }
}

float* input_concat;
float* linear_result;
uint32_t current_layer;
uint32_t current_step;

void concat_in_h(){
    for (uint32_t x = 0; x < hiddenUnites; x++) {
        float inVal = rsGetElementAt_float(inputs, x, current_step);
        input_concat[x] = inVal;
        input_concat[hiddenUnites + x] = h[x];
    }
}

float RS_KERNEL linear_map(uint32_t x){
    float sum = 0.0f;
    //rsDebug("current_layer=",  current_layer);
    for (uint32_t unit = 0; unit < hiddenUnites * 2; unit++) {
        float a = input_concat[unit];
        float b = rsGetElementAt_float(weights, x, unit, current_layer);
        //rsDebug("a=",  (*a));
        //rsDebug("b=",  (*b));
        sum += a * b;
        //rsDebug("sum=",  sum);
    }

    float valB = rsGetElementAt_float(biases, x, current_layer);
    return sum + valB;
}

void linear_map_func(){
    for (uint32_t x = 0; x < hiddenUnites * 4; x++) {
        float sum = 0.0f;
        for (uint32_t unit = 0; unit < hiddenUnites * 2; unit++) {
            float a = input_concat[unit];
            float b = rsGetElementAt_float(weights, x, unit, current_layer);
            //rsDebug("a=",  (*a));
            //rsDebug("b=",  (*b));
            sum += a * b;
            //rsDebug("sum=",  sum);
        }
        float valB = rsGetElementAt_float(biases, x, current_layer);
        linear_result[x] = sum + valB;
   }
}

static inline float sigmoid(float x){
     return 1.0f / (1.0f + exp(-x));
}

void RS_KERNEL pointwise_ch(float in, uint32_t x){
    float cVal = c[x];
    float fVal = linear_result[hiddenUnites * 2 + x];
    float iVal = linear_result[x];
    float jVal = linear_result[hiddenUnites + x];
    float oVal = linear_result[hiddenUnites * 3 + x];
    c[x] = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanh(jVal);
    //rsDebug("newC:", newC);
    h[x] = tanh(cVal) * sigmoid(oVal);
}

void pointwise_ch_func(){
    for (uint32_t x = 0; x < hiddenUnites; x++) {
        float cVal = c[x];
        float fVal = linear_result[hiddenUnites * 2 + x];
        float iVal = linear_result[x];
        float jVal = linear_result[hiddenUnites + x];
        float oVal = linear_result[hiddenUnites * 3 + x];
        c[x] = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanh(jVal);
        h[x] = tanh(cVal) * sigmoid(oVal);
    }
}

void RS_KERNEL update_input(float in, uint32_t x){
    rsSetElementAt_float(inputs, in, x, current_step);
}

void update_input_func(){
    for (uint32_t x = 0; x < hiddenUnites; x++) {
        rsSetElementAt_float(inputs, h[x], x, current_step);
    }
}

void calc_cell_one_step(){
    for (uint32_t x = 0; x < hiddenUnites; x++) {
        float inVal = rsGetElementAt_float(inputs, x, current_step);
        input_concat[x] = inVal;
        input_concat[hiddenUnites + x] = h[x];
    }

    for (uint32_t x = 0; x < hiddenUnites * 4; x++) {
        float sum = 0.0f;
        for (uint32_t unit = 0; unit < hiddenUnites * 2; unit++) {
            float a = input_concat[unit];
            float b = rsGetElementAt_float(weights, x, unit, current_layer);
            //rsDebug("a=",  (*a));
            //rsDebug("b=",  (*b));
            sum += a * b;
            //rsDebug("sum=",  sum);
        }
        float valB = rsGetElementAt_float(biases, x, current_layer);
        linear_result[x] = sum + valB;
   }

    for (uint32_t x = 0; x < hiddenUnites; x++) {
        float cVal = c[x];
        float fVal = linear_result[hiddenUnites * 2 + x];
        float iVal = linear_result[x];
        float jVal = linear_result[hiddenUnites + x];
        float oVal = linear_result[hiddenUnites * 3 + x];
        c[x] = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanh(jVal);
        h[x] = tanh(cVal) * sigmoid(oVal);
    }

    for (uint32_t x = 0; x < hiddenUnites; x++) {
        rsSetElementAt_float(inputs, h[x], x, current_step);
    }
}

rs_allocation w_out;
rs_allocation b_out;

float* label_prob;
uint32_t outDim;

float RS_KERNEL output_transform(uint32_t x){
    float sum = 0.0f;
    for (uint32_t unit = 0; unit < hiddenUnites; unit++) {
        float a = h[unit];
        float b = rsGetElementAt_float(w_out, x, unit);
        sum += a * b;
    }
    float valB = rsGetElementAt_float(b_out, x);
    sum += valB;
    return sum;
}

void output_transform_func(){
    for (uint32_t x = 0; x < outDim; x++) {
        float sum = 0.0f;
        for (uint32_t unit = 0; unit < hiddenUnites; unit++) {
            float a = h[unit];
            float b = rsGetElementAt_float(w_out, x, unit);
            sum += a * b;
        }
        float valB = rsGetElementAt_float(b_out, x);
        label_prob[x] = sum + valB;
   }
}

void all_in_one(){
    input_transform_func();
    for (uint32_t layer = 0; layer < layerSize; layer++) {
        set_ch_zeros();
        current_layer = layer;
        for (uint32_t t = 0; t < timeSteps; t++) {
            current_step =t;
            calc_cell_one_step();
        }
   }
   output_transform_func();
}
