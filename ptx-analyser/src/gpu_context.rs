#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Architecture {
    Kepler,
    Maxwell,
    Pascal,
    Volta,
    Turing,
    Ampere,
    AdaLovelace,
    Hopper,
    Blackwell,
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct GpuTopology {
    pub architecture: Architecture,
    pub compute_capability: (u8, u8),
    pub boost_clock_mhz: u32,
    pub tdp_watts: u32,

    //cluster hierarchy: chip = gpcs x tpcs_per_gpc x sms_per_tpc
    pub gpcs: u32,
    pub tpcs_per_gpc: u32,
    pub sms_per_tpc: u32,
    pub total_sms: u32,

    //per-SM compute units
    pub cuda_cores_per_sm: u32,
    //Ada splits the 128 cores: 64 FP32-only and 64 FP32/INT32
    pub fp32_only_lanes: u32,
    pub fp32_int32_lanes: u32,
    pub tensor_core_gen: u8,
    //2 per SM on Ada present for correctness, not performance.
    //fp64 throughput is 1/64th of fp32.
    pub fp64_cores_per_sm: u32,
    //one per SM partition, 4 total on Ada
    pub tensor_cores_per_sm: u32,
    //one 3rd-gen RT core per SM on Ada
    pub rt_cores_per_sm: u32,
    //one per SM partition
    pub tex_units_per_sm: u32,
    pub warp_schedulers_per_sm: u32,
    //Ada dual-dispatches, so 2 per scheduler = 8
    pub dispatch_units_per_sm: u32,
    pub load_store_units_per_sm: u32,
    //handle transcendental ops (sin, cos, rcp, rsqrt, ex2) that can't go through
    //the regular FP32 pipeline. kernels heavy on these can become SFU-bound.
    pub sfus_per_sm: u32,
    pub max_warps_per_sm: u32,
    pub max_threads_per_sm: u32,
    pub max_threads_per_block: u32,

    //register file register pressure determines occupancy.
    //registers are 32-bit but a kernel may use 64-bit (2 regs) or 16-bit (half a reg).
    //regs_per_sm counts 32-bit slots; adjust regs_per_thread accordingly.
    pub regs_per_sm: u32,
    pub reg_file_bytes_per_sm: u32,
    //hard limit per thread, 255 on Kepler+
    pub max_regs_per_thread: u32,

    //per-SM cache. L1 and shared memory share the same physical pool;
    //the split is chosen at runtime via cudaFuncSetAttribute.
    //instruction cache, Volta+
    pub l0_icache_bytes: Option<u32>,
    pub l1_shared_pool_bytes: u32,
    pub max_smem_per_block_bytes: u32,

    //chip-wide
    pub total_cuda_cores: u32,
    pub total_tensor_cores: u32,
    //shared across all SMs; Ada L2 is 16x larger than Ampere
    pub l2_bytes: u32,

    //off-chip GDDR memory
    pub vram_gb: u32,
    pub bus_width_bits: u32,
    pub memory_controllers: u32,
    pub bandwidth_gb_s: u32,
    pub memory_clock_gbps: f32,

    //peak throughput at boost clock
    pub fp64_tflops: f32,
    pub fp32_tflops: f32,
    pub fp16_tflops: f32,
    pub tensor_fp16_tflops: f32,
    pub tensor_bf16_tflops: f32,
    pub tensor_tf32_tflops: f32,
    pub tensor_fp8_tflops: f32,
    pub tensor_int8_tops: f32,
}

impl GpuTopology {
    pub fn total_register_file_bytes(&self) -> u32 {
        self.total_sms * self.reg_file_bytes_per_sm
    }

    //max warps resident on one SM given a kernel's register usage.
    //regs_per_thread should be in 32-bit register slots use 2 per thread for f64,
    //or 0.5 (round up to 1) for f16 if the compiler doesn't pack them.
    pub fn max_resident_warps(&self, regs_per_thread: u32) -> Option<u32> {
        if regs_per_thread == 0 { return None; }
        Some(self.regs_per_sm / (regs_per_thread * 32))
    }

    //occupancy as a fraction [0, 1]: ratio of resident warps to the SM maximum
    pub fn occupancy(&self, regs_per_thread: u32) -> Option<f32> {
        self.max_resident_warps(regs_per_thread)
            .map(|w| (w.min(self.max_warps_per_sm) as f32) / self.max_warps_per_sm as f32)
    }

    //fraction of the register file in use given active warps and regs/thread
    pub fn reg_file_utilisation(&self, regs_per_thread: u32, active_warps: u32) -> f32 {
        (regs_per_thread * 32 * active_warps) as f32 / self.regs_per_sm as f32
    }

    //active threads across the whole chip at a given occupancy fraction
    pub fn active_threads(&self, occupancy: f32) -> u32 {
        let warps_per_sm = (self.max_warps_per_sm as f32 * occupancy).round() as u32;
        self.total_sms * warps_per_sm * 32
    }

    //arithmetic intensity (FLOP/byte) above which a kernel is compute-bound.
    //pass fp32_tflops, tensor_fp16_tflops etc. to get the ridge for each precision.
    pub fn roofline_ridge_point(&self, flops: f32) -> f32 {
        flops * 1e12 / (self.bandwidth_gb_s as f32 * 1e9)
    }

    //theoretical peak FP32 TFLOPS derived from core count and clock.
    //each CUDA core issues 1 FMA (2 FLOP) per clock.
    pub fn peak_fp32_tflops_derived(&self) -> f32 {
        (self.total_cuda_cores as f64 * 2.0 * self.boost_clock_mhz as f64 * 1e6 / 1e12) as f32
    }

    //how many thread blocks fit on one SM given shared memory usage per block
    pub fn max_blocks_per_sm_smem(&self, smem_per_block_bytes: u32) -> u32 {
        if smem_per_block_bytes == 0 { return self.max_warps_per_sm; }
        self.l1_shared_pool_bytes / smem_per_block_bytes
    }

    //RTX 4080 Super — full AD103, 80 SMs (sm_89)
    pub fn rtx_4080_super() -> Self {
        let sms = 80;
        let regs_per_sm = 65_536;

        Self {
            architecture: Architecture::AdaLovelace,
            compute_capability: (8, 9),
            boost_clock_mhz: 2550,
            tdp_watts: 320,

            gpcs: 7,
            tpcs_per_gpc: 6,
            sms_per_tpc: 2,
            total_sms: sms,

            cuda_cores_per_sm: 128,
            fp32_only_lanes: 64,
            fp32_int32_lanes: 64,
            tensor_core_gen: 4,
            fp64_cores_per_sm: 2,
            tensor_cores_per_sm: 4,
            rt_cores_per_sm: 1,
            tex_units_per_sm: 4,
            warp_schedulers_per_sm: 4,
            dispatch_units_per_sm: 8,
            load_store_units_per_sm: 16,
            sfus_per_sm: 4,
            max_threads_per_sm: 1536,
            max_threads_per_block: 1024,

            max_warps_per_sm: 48, // 1536 max_threads_per_sm / 32 threads_per_warp
            regs_per_sm,
            reg_file_bytes_per_sm: regs_per_sm * 4,
            max_regs_per_thread: 255,

            l0_icache_bytes: Some(32 * 1024),
            l1_shared_pool_bytes: 128 * 1024,
            //may be set, use cudaDevAttrMaxSharedMemoryPerBlockOptin to query the actual per-block limit.
            max_smem_per_block_bytes: 99 * 1024,

            total_cuda_cores:   sms * 128,
            total_tensor_cores: sms * 4,
            l2_bytes:           64 * 1024 * 1024,

            vram_gb: 16,
            bus_width_bits: 256,
            memory_controllers: 8,
            bandwidth_gb_s: 736,
            memory_clock_gbps: 23.0,

            //very much theoretical peaks assuming 100% utilization of all lanes.
            fp64_tflops:        52.2 / 64.0,
            fp32_tflops:        52.2,
            fp16_tflops:        52.2,
            tensor_fp16_tflops: 208.0,
            tensor_bf16_tflops: 208.0,
            tensor_tf32_tflops: 104.0,
            tensor_fp8_tflops:  416.0,
            tensor_int8_tops:   416.0,
        }
    }
}