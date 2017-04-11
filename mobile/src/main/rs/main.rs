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
//rs_allocation outputs;

float RS_KERNEL set_zeros(uint32_t x, uint32_t y){
    return 0.0f;
}

static void print(rs_allocation mat){
    const uint32_t dimX =  rsAllocationGetDimX(mat);
    const uint32_t dimY =  rsAllocationGetDimY(mat);
    rsDebug("dimX: ", dimX);
    rsDebug("dimY: ", dimY);
    for (uint32_t x = 0; x < dimX; x++){
        for (uint32_t y = 0; y < dimY; y++){
            float a = rsGetElementAt_float(mat, x, y);
            rsDebug("x: ", x);
            rsDebug("y: ", y);
            rsDebug("mat: ", a);
         }     
    }
}

rs_allocation c;
rs_allocation h;

rs_allocation input_concat;
rs_allocation linear_result;
uint32_t current_layer;
uint32_t current_step;

void RS_KERNEL concat_in(float in, uint32_t x, uint32_t y){
    float inputVal = rsGetElementAt_float(inputs, x, current_step);
    rsSetElementAt_float(input_concat, inputVal, x);
}

void RS_KERNEL concat_h(float in, uint32_t x){
    rsSetElementAt_float(input_concat, in, hiddenUnites + x);
}

float RS_KERNEL linear_map(uint32_t x){
    float sum = 0.0f;
    //rsDebug("current_layer=",  current_layer);
    for (uint32_t c = 0; c < hiddenUnites * 2; c++) {
        float a = rsGetElementAt_float(input_concat, c);
        float b = rsGetElementAt_float(weights, x, c, current_layer);
        //rsDebug("a=",  (*a));
        //rsDebug("b=",  (*b));
        sum += a * b;
        //rsDebug("sum=",  sum);
    }

    float valB = rsGetElementAt_float(biases, x, current_layer);
    sum += valB;
    return sum;
}

rs_allocation i_gate;
rs_allocation j_val; // C_hat in original formula
rs_allocation f_gate;
rs_allocation o_gate;

float RS_KERNEL get_i(uint32_t x){
    return rsGetElementAt_float(linear_result, x);
}

float RS_KERNEL get_j(uint32_t x){
    return rsGetElementAt_float(linear_result, hiddenUnites + x);
}

float RS_KERNEL get_f(uint32_t x){
    return rsGetElementAt_float(linear_result, hiddenUnites * 2 + x);
}

float RS_KERNEL get_o(uint32_t x){
    return rsGetElementAt_float(linear_result, hiddenUnites * 3 + x);
}

static inline float sigmoid(float x){
     return 1.0f / (1.0f + exp(-x));
}

float RS_KERNEL pointwise_c(uint32_t x){
    float cVal = rsGetElementAt_float(c, x);
    float fVal = rsGetElementAt_float(f_gate, x);
    float iVal = rsGetElementAt_float(i_gate, x);
    float jVal = rsGetElementAt_float(j_val, x);

    float newC = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanh(jVal);
    //rsDebug("newC:", newC);
    return newC;
}

float RS_KERNEL pointwise_h(uint32_t x){
    float cVal = rsGetElementAt_float(c, x);
    float oVal = rsGetElementAt_float(o_gate, x);

    return tanh(cVal) * sigmoid(oVal);
}

void RS_KERNEL update_input(float in, uint32_t x){
    rsSetElementAt_float(inputs, in, x, current_step);
}

// float RS_KERNEL set_output(float in, uint32_t x){
//     return in;
// }

rs_allocation w_out;
rs_allocation b_out;

rs_allocation label_prob;

float RS_KERNEL output_transform(uint32_t x){
    float sum = 0.0f;
    for (uint32_t c = 0; c < hiddenUnites; c++) {
        float a = rsGetElementAt_float(h, c);
        float b = rsGetElementAt_float(w_out, x, c);
        sum += a * b;
    }
    float valB = rsGetElementAt_float(b_out, x);
    sum += valB;
    return sum;
}

