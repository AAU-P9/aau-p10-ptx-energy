__global__ void test(long long *sink) {
    long long p;
    asm volatile("cvta.shared.u64 %0, %1;" : "=l"(p) : "l"((long long)threadIdx.x));
}
