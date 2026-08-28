// Live occupancy probe for the crossover kernel class (operator directive:
// verify limits on silicon, never assume). Reports, for a 2-CTA cluster kernel
// with the crossover's exact footprint (256 threads, ~100,608 B dynamic smem):
//   - cudaOccupancyMaxActiveBlocksPerMultiprocessor
//   - cudaOccupancyMaxActiveClusters (the cluster-scheduling reality: GPC
//     packing can cap concurrent clusters BELOW SMs/2, silently serializing
//     cluster waves -- a candidate contributor to the CFG=3 slowdown)
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(call)                                                                  \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            printf("[FAILED] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            return 1;                                                             \
        }                                                                         \
    } while (0)

extern __shared__ char smem_raw[];

__global__ void __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
probe_cluster_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = smem_raw[0];
}

__global__ void __launch_bounds__(256, 1)
probe_plain_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = smem_raw[0];
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);

    const size_t SMEMS[] = { 100608, 98816, 95232 };
    const char* NAMES[] = { "crossover(100608B)", "dense-bm64(98816B)", "dense-m16(95232B)" };

    for (int i = 0; i < 3; ++i) {
        CK(cudaFuncSetAttribute(probe_plain_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEMS[i]));
        int blocks = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, probe_plain_kernel,
                                                          256, SMEMS[i]));
        printf("plain   %s: blocks/SM = %d -> max concurrent CTAs = %d\n",
               NAMES[i], blocks, blocks * prop.multiProcessorCount);
    }

    CK(cudaFuncSetAttribute(probe_cluster_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 100608));
    {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(188, 1, 1);      // even; representative of the CFG=3 grid
        cfg.blockDim = dim3(256, 1, 1);
        cfg.dynamicSmemBytes = 100608;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = 2; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
        cfg.attrs = attrs; cfg.numAttrs = 1;
        int clusters = 0;
        CK(cudaOccupancyMaxActiveClusters(&clusters, probe_cluster_kernel, &cfg));
        printf("cluster crossover(100608B): max ACTIVE clusters = %d (= %d CTAs of %d SMs)\n",
               clusters, clusters * 2, prop.multiProcessorCount);
    }
    printf("OCCUPANCY_PROBE_DONE\n");
    return 0;
}
