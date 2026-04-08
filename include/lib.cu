/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <thread>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include "helper_cupti.h"
#include "range_profiling.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Tracy headers
#include "../../vendor/libEPMD/libEMPD.h"
#include "tracy/Tracy.hpp"
#include "tracy/TracyCUDA.hpp"

#include "../nvml_clock_control/include/NvmlClockControl.hpp"

// NVML headers
#include <nvml.h>

const char *const sl_PowerWattageNVMLPeriodicSampling =
    "Power (mW) - NVML Periodic Sampling";
const char *const sl_TemperatureNVMLPeriodicSampling =
    "Temperature (C) - NVML Periodic Sampling";
const char *const sl_PowerWattagePMD2SlotPeriodicSampling =
    "Power (mW) - PMD2 Periodic Sampling (Slot)";
const char *const sl_PowerWattagePMD2CablePeriodicSampling =
    "Power (mW) - PMD2 Periodic Sampling (Cable)";

const char *const sl_GPUClock = "GPU Clock (MHz)";
const char *const sl_MemoryClock = "GPU Memory Clock (MHz)";

std::atomic<bool> g_Running{true};
std::atomic<bool> g_MonitoringReady{false};
unsigned int nvml_temp = 0;
tracy::CUDACtx *tracyCtx_;

bool isEnvironmentVariableEnabled(const char *varName) {
  const char *value = std::getenv(varName);
  bool enabled = (value != nullptr) &&
                 (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0);
  printf("Environment Variable %s is %s\n", varName,
         enabled ? "ENABLED" : "DISABLED");

  return enabled;
}

bool EnvironmentVariableHasValue(const char *varName) {
  const char *value = std::getenv(varName);
  bool hasValue =
      (value != nullptr) && (strlen(value) > 0) && (strcmp(value, "None") != 0);

  printf("Environment Variable %s has value: %s\n", varName,
         hasValue ? value : "NULL");

  return hasValue;
}

bool isProfilingEnabled =
    isEnvironmentVariableEnabled("BENCHMARK_PROFILING_ENABLED");
bool isPMD2Enabled = isEnvironmentVariableEnabled("BENCHMARK_PMD2_ENABLED");
bool isNvmlPowerMonitoringEnabled =
    isEnvironmentVariableEnabled("BENCHMARK_NVML_POWER_MONITORING_ENABLED");

bool isGpuClockMinEnabled =
    EnvironmentVariableHasValue("BENCHMARK_GPU_CLOCK_MIN");
bool isGpuClockMaxEnabled =
    EnvironmentVariableHasValue("BENCHMARK_GPU_CLOCK_MAX");
bool isMemoryClockMinEnabled =
    EnvironmentVariableHasValue("BENCHMARK_MEMORY_CLOCK_MIN");
bool isMemoryClockMaxEnabled =
    EnvironmentVariableHasValue("BENCHMARK_MEMORY_CLOCK_MAX");

void sensor_data_callback(uint64_t timestamp_us,
                          const pmd2_sensor_data_t *sensor_data) {
  TracyPlot(sl_PowerWattagePMD2SlotPeriodicSampling,
            static_cast<int64_t>(sensor_data->power_readings[7].power));
  TracyPlot(sl_PowerWattagePMD2CablePeriodicSampling,
            static_cast<int64_t>(sensor_data->power_readings[4].power));
}

void gpuPowerMonitorThread() {
  nvmlDevice_t device;
  nvmlReturn_t result;

  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(result));
    g_MonitoringReady.store(true); // Signal ready even on failure
    return;
  }

  result = nvmlDeviceGetHandleByIndex(0, &device);
  if (result != NVML_SUCCESS) {
    fprintf(stderr, "Failed to get handle for GPU 0: %s\n",
            nvmlErrorString(result));
    nvmlShutdown();
    g_MonitoringReady.store(true); // Signal ready even on failure
    return;
  }

  // Signal that monitoring is ready
  g_MonitoringReady.store(true);

  unsigned int power_mW = 0;
  while (g_Running.load()) {
    // Read the power usage
    result = nvmlDeviceGetPowerUsage(device, &power_mW);
    if (result == NVML_SUCCESS) {
      TracyPlot(sl_PowerWattageNVMLPeriodicSampling,
                static_cast<int64_t>(power_mW));
    }

    // Read the temperature
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &nvml_temp);
    if (result == NVML_SUCCESS) {
      TracyPlot(sl_TemperatureNVMLPeriodicSampling,
                static_cast<int64_t>(nvml_temp));
    }

    // Sample every 100ms (adjust as needed)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  nvmlShutdown();
}

RangeProfilerTargetPtr pRangeProfilerTarget;

std::vector<const char *> metrics = {
    "sm__ctas_launched.sum",
    "sm__warps_active.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "l1tex__t_requests_pipe_tex_mem_texture.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "dram__bytes_read.sum.per_second",
    "dram__sectors_read.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum.per_second",
    "dram__sectors_write.sum",
    "smsp__warps_eligible.sum.per_cycle_active",
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_"
    "peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.avg.pct_of_"
    "peak_sustained_elapsed",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second",
    "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_control_pred_on.sum",
    "smsp__sass_inst_executed_op_global_atom.sum",
    "smsp__inst_executed_op_global_ld.sum",
    "smsp__inst_executed_op_global_st.sum",
    "smsp__inst_executed_op_global_red.sum",
    "smsp__inst_executed_op_shared_atom.sum",
    "smsp__inst_executed_op_shared_atom_dot_alu.sum",
    "smsp__inst_executed_op_shared_atom_dot_cas.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "smsp__inst_executed_op_surface_atom.sum",
    "smsp__inst_executed_op_texture.sum",
    "smsp__inst_executed_op_surface_ld.sum",
    "smsp__inst_executed_op_surface_st.sum",
    "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum",
    "smsp__inst_issued.sum",
    "smsp__sass_thread_inst_executed_op_misc_pred_on.sum",
    "smsp__average_inst_executed_per_warp.ratio",
    "lts__t_sectors_op_read.sum.per_second",
    "lts__t_sectors_srcunit_tex_op_read.sum.per_second",
    "lts__t_sector_op_read_hit_rate.pct",
    "lts__t_sector_op_write_hit_rate.pct",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_launched.sum",
    "smsp__thread_inst_executed.sum",
    "smsp__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_issued.avg.per_cycle_active",
    "smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_"
    "elapsed",
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__texin_sm2tex_req_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_misc_per_warp_active.pct",
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
    "lts__t_sectors_aperture_sysmem_op_read.sum.per_second",
    "lts__t_sectors_aperture_sysmem_op_write.sum.per_second",
    "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active",
    "l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",
    "l1tex__f_tex2sm_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__thread_inst_executed_per_inst_executed.pct",
};

CuptiProfilerHostPtr pCuptiProfilerHost;
std::vector<uint8_t> configImage;
std::vector<uint8_t> counterDataImage;
std::thread gpuMonitorThread; // Don't start the thread here - wait for setup()
CUcontext cuContext;

void ProfilingDeviceSupportStatus(CUdevice device) {
  CUpti_Profiler_DeviceSupported_Params params = {
      CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
  params.cuDevice = device;
  params.api = CUPTI_PROFILER_RANGE_PROFILING;
  CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

  if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    ::std::cerr << "Unable to profile on device " << device << ::std::endl;

    if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
    }

    if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tdevice sli configuration is not supported"
                  << ::std::endl;
    }

    if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tdevice vgpu configuration is not supported"
                  << ::std::endl;
    } else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED) {
      ::std::cerr << "\tdevice vgpu configuration disabled profiling support"
                  << ::std::endl;
    }

    if (params.confidentialCompute ==
        CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr
          << "\tdevice confidential compute configuration is not supported"
          << ::std::endl;
    }

    if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported"
                  << ::std::endl;
    }

    if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tWSL is not supported" << ::std::endl;
    }

    exit(EXIT_WAIVED);
  }
}

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (isnan(diff) || diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void profileProgramCounters(void (*func)()) {
  pmd2_device_t device = {0};

  if (isPMD2Enabled) {
    // Start the power consumption monitoring (Physical)
    pmd2_config_t pmd_config = PMD2_DEVICE;
    pmd2_port_info_t ports[PMD2_MAX_PORTS];

    int foundDevices = pmd2_find_devices(ports, PMD2_MAX_PORTS, &pmd_config);

    int ret = pmd2_open(&device, ports[0].port_name, &pmd_config);

    pmd2_thread_data_t *thread_data =
        (pmd2_thread_data_t *)malloc(sizeof(pmd2_thread_data_t));
    *thread_data = (pmd2_thread_data_t){0};

    pmd2_read_continuously(&device, sensor_data_callback, 750, thread_data);
  }

  if (isNvmlPowerMonitoringEnabled) {
    // Start the power consumption monitoring thread
    g_Running.store(true);
    g_MonitoringReady.store(false);
    gpuMonitorThread = std::thread(gpuPowerMonitorThread);

    // Wait for the monitoring thread to be ready
    while (!g_MonitoringReady.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
/*
  NvmlClockControl nvmlClockControl;
  if (isGpuClockMinEnabled && isGpuClockMaxEnabled) {

    unsigned int gfxMax = nvmlClockControl.getMaxGraphicsClock();
    unsigned int clockMin = std::atol(std::getenv("BENCHMARK_GPU_CLOCK_MIN"));
    unsigned int clockMax = std::atol(std::getenv("BENCHMARK_GPU_CLOCK_MAX"));

    if (clockMin > clockMax) {
      std::cerr << "BENCHMARK_GPU_CLOCK_MIN cannot be greater than "
                   "BENCHMARK_GPU_CLOCK_MAX"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    if (clockMax > gfxMax) {
      std::cerr << "BENCHMARK_GPU_CLOCK_MAX cannot be greater than the maximum "
                   "graphics clock of the GPU: "
                << gfxMax << " MHz" << std::endl;
      exit(EXIT_FAILURE);
    }

    printf("Setting GPU clocks to min %u max %u\n", clockMin, clockMax);

    nvmlClockControl.setGPUClocks(clockMin, clockMax);
  }

  if (isMemoryClockMinEnabled && isMemoryClockMaxEnabled) {
    unsigned int memMax = nvmlClockControl.getMaxMemoryClock();
    unsigned int clockMin =
        isMemoryClockMinEnabled
            ? std::atol(std::getenv("BENCHMARK_MEMORY_CLOCK_MIN"))
            : 0;
    unsigned int clockMax =
        isMemoryClockMaxEnabled
            ? std::atol(std::getenv("BENCHMARK_MEMORY_CLOCK_MAX"))
            : 0;
    if (clockMin > clockMax) {
      std::cerr << "BENCHMARK_MEMORY_CLOCK_MIN cannot be greater than "
                   "BENCHMARK_MEMORY_CLOCK_MAX"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    if (clockMax > memMax) {
      std::cerr << "BENCHMARK_MEMORY_CLOCK_MAX cannot be greater than the "
                   "maximum memory clock of the GPU: "
                << memMax << " MHz" << std::endl;
      exit(EXIT_FAILURE);
    }
    printf("Setting memory clocks to min %u max %u\n", clockMin, clockMax);

    nvmlClockControl.setGPUMemoryClocks(clockMin, clockMax);
  }

*/

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Setup Tracy plots. Start Tracy CUDA profiling only when CUPTI range
  // profiling is also enabled to avoid creating CUDA/CUPTI activity without the
  // profiler.
  TracyPlotConfig(sl_PowerWattagePMD2CablePeriodicSampling,
                  tracy::PlotFormatType::Number, false, true,
                  tracy::Color::LightBlue1);
  TracyPlotConfig(sl_PowerWattagePMD2SlotPeriodicSampling,
                  tracy::PlotFormatType::Number, false, true,
                  tracy::Color::IndianRed1);
  TracyPlotConfig(sl_PowerWattageNVMLPeriodicSampling,
                  tracy::PlotFormatType::Number, false, true,
                  tracy::Color::Chocolate3);
  TracyPlotConfig(sl_TemperatureNVMLPeriodicSampling,
                  tracy::PlotFormatType::Number, false, true,
                  tracy::Color::Cornsilk4);

  //(TODO) maybe should plot this
  TracyPlotConfig(sl_GPUClock, tracy::PlotFormatType::Number, false, true,
                  tracy::Color::Cyan1);
  TracyPlotConfig(sl_MemoryClock, tracy::PlotFormatType::Number, false, true,
                  tracy::Color::Green1);


  tracyCtx_ = TracyCUDAContext();
  TracyCUDAStartProfiling(tracyCtx_);

  printf("isGpuClockMinEnabled %d isGpuClockMaxEnabled %d "
         "isMemoryClockMinEnabled %d isMemoryClockMaxEnabled %d\n",
         isGpuClockMinEnabled, isGpuClockMaxEnabled, isMemoryClockMinEnabled,
         isMemoryClockMaxEnabled);

  if (isProfilingEnabled) {
    std::cout << "Profiling is ENABLED" << std::endl;
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, 0));

    // Get the current ctx for the device
    ProfilingDeviceSupportStatus(cuDevice);

    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGetAttribute(
        &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(
        &computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cuDevice));

    if (computeCapabilityMajor < 7 ||
        (computeCapabilityMajor == 7 && computeCapabilityMinor < 5)) {
      std::cerr << "Range Profiling is supported only on devices with compute "
                   "capability 7.5 and above"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    RangeProfilerConfig config;
    config.maxNumOfRanges = 20;
    config.minNestingLevel = 1;
    config.numOfNestingLevel = 1;

    pCuptiProfilerHost = std::make_shared<CuptiProfilerHost>();

    // Create a context
    // DRIVER_API_CALL(cuCtxCreate(&cuContext, (CUctxCreateParams *)0, 0,
    // cuDevice));
    DRIVER_API_CALL(cuCtxGetCurrent(&cuContext));
    pRangeProfilerTarget =
        std::make_shared<RangeProfilerTarget>(cuContext, config);

    // Get chip name
    std::string chipName;
    CUPTI_API_CALL(RangeProfilerTarget::GetChipName(cuDevice, chipName));

    // Get Counter availability image
    std::vector<uint8_t> counterAvailabilityImage;
    CUPTI_API_CALL(RangeProfilerTarget::GetCounterAvailabilityImage(
        cuContext, counterAvailabilityImage));

    // Create config image
    pCuptiProfilerHost->SetUp(chipName, counterAvailabilityImage);
    CUPTI_API_CALL(pCuptiProfilerHost->CreateConfigImage(metrics, configImage));

    CUPTI_API_CALL(pRangeProfilerTarget->EnableRangeProfiler());

    // Create CounterData Image
    CUPTI_API_CALL(pRangeProfilerTarget->CreateCounterDataImage(
        metrics, counterDataImage));

    // Set range profiler configuration
    CUPTI_API_CALL(pRangeProfilerTarget->SetConfig(
        CUPTI_UserRange, CUPTI_UserReplay, configImage, counterDataImage));

    // Run the target function
    while (!pRangeProfilerTarget->IsAllPassSubmitted()) {
      CUPTI_API_CALL(pRangeProfilerTarget->StartRangeProfiler());

      CUPTI_API_CALL(pRangeProfilerTarget->PushRange("MainRange"));

      func();

      cudaError_t err = cudaGetLastError();

      if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

      CUPTI_API_CALL(pRangeProfilerTarget->PopRange());

      // Stop Range Profiling (this returns whether all passes are submitted)
      CUPTI_API_CALL(pRangeProfilerTarget->StopRangeProfiler());

      // This is used to line up the slicing of the power curve with the
      // profiling ranges See merge.py script for details
      TracyMessageL("Range Profiler Pass Completed");
    };

    // Ensure all preceding CUDA calls are complete
    // RUNTIME_API_CALL(cudaGetLastError());
    // RUNTIME_API_CALL(cudaDeviceSynchronize());

    // Get Profiler Data
    CUPTI_API_CALL(pRangeProfilerTarget->DecodeCounterData());

    // Evaluate the results
    size_t numRanges = 0;
    CUPTI_API_CALL(
        pCuptiProfilerHost->GetNumOfRanges(counterDataImage, numRanges));
    printf("Number of Ranges: %zu\n", numRanges);
    for (size_t rangeIndex = 0; rangeIndex < numRanges; ++rangeIndex) {
      CUPTI_API_CALL(pCuptiProfilerHost->EvaluateCounterData(
          rangeIndex, metrics, counterDataImage));
    }

    pCuptiProfilerHost->PrintProfilerRanges();

    CUPTI_API_CALL(pRangeProfilerTarget->DisableRangeProfiler());
    pCuptiProfilerHost->TearDown();
  } else {
    // Just run the target function without profiling
    func();

    // Better detect asynchronous kernel failures by synchronizing first.
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
      fprintf(stderr, "Kernel execution failed (synchronize): %s\n",
              cudaGetErrorString(syncErr));
      exit(EXIT_FAILURE);
    }

    // Clear any prior error and keep behavior similar to previous code.
    cudaGetLastError();

    TracyMessageL("Range Profiler Pass Completed");
  }

  // Sleep for 500 ms, to ensure NVML has a sample on the other side of the
  // kernel execution
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Stop the GPU monitoring thread
  if (isNvmlPowerMonitoringEnabled) {
    g_Running.store(false);
    gpuMonitorThread.join();
  }

  // Stop PMD2 power monitoring
  if (isPMD2Enabled) {
    pmd2_stop_continuous_reading(&device);
  }

  if (isProfilingEnabled && tracyCtx_ != nullptr) {
    // Clean up Tracy CUDA context only if we started it earlier.
    TracyCUDACollect(tracyCtx_);
    TracyCUDAStopProfiling(tracyCtx_);
    // TracyCUDAContextDestroy(tracyCtx_); // Disabled: sometimes causes
    // crash/shutdown
  }
}
