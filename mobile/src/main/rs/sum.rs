// Needed directive for RS to work
#pragma version(1)

// Change java_package_name directive to match your Activity's package path
#pragma rs java_package_name(net.hydex11.firstexample)

// This kernel function will just sum 2 to every input element
// * in -> Current Allocation element
// * x  -> Current element index
int __attribute__((kernel)) sum2(int in, uint32_t x) {

    // Performs the sum and returns the new value
    return in + 2;

}