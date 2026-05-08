use std::collections::{HashMap, HashSet};
use serde::Serialize;

use ptx_parser::r#type::{
    FunctionStatement,
    instruction::{
        Inst,
        ld::section_0::Ss as LdSs,
        st::section_0::Ss as StSs,
        atom::section_0::Space as AtomSpace,
        red::section_0::Space as RedSpace0,
        red::section_1::Space as RedSpace1,
        prefetch::section_0::Space as PrefetchSpace,
    },
};

use super::common::{BlockId, ControlFlowGraph, statement_opcode};
use super::ddg::{DataDependencyGraph, InstrId};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How the address of a memory instruction was derived.
/// Determines cache behaviour: coalescing, reuse distance, miss rate.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum AddressPattern {
    /// Address is a linear function of threadIdx/blockIdx.
    /// Consecutive threads address consecutive words → warp coalesces to one transaction.
    TidLinear,
    /// Address depends on a value loaded from global memory (pointer chase).
    /// Almost always misses every level of cache.
    Indirect,
    /// Address does not change across loop iterations (loop-invariant).
    /// After the first access, subsequent accesses are L1/L2 hits.
    Invariant,
    /// Address changes per loop iteration through an induction variable,
    /// not tied to thread ID. Strides through memory → working-set grows with iters.
    LoopVariable,
    /// Could not classify with the heuristics available.
    Complex,
}

/// Per-occurrence memory access profile for one instruction instance.
/// "Occurrence" here means a specific (block_id, instr_idx) site in the CFG,
/// not just an opcode — two `ld.global.f32` in different blocks get separate profiles.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryProfile {
    pub block_id: usize,
    pub instr_idx: usize,
    pub opcode: String,
    /// Weighted occurrence count (same scale as instructionOccurrences).
    pub occurrences: u64,
    pub loop_depth: usize,
    pub address_pattern: AddressPattern,
    /// Fraction of accesses estimated to miss L1 (0.0 = always hits, 1.0 = always misses).
    pub estimated_miss_rate: f64,
    /// Estimated additional pipeline stall cycles per access (rough model).
    pub estimated_stall_cycles: f64,
    /// Total estimated stall cycles weighted by occurrences.
    pub estimated_total_stall_cycles: f64,
}

// ---------------------------------------------------------------------------
// Memory state space
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum MemSpace {
    Global,
    Shared,
    Other,
}

fn ld_ss(ss: &Option<LdSs>) -> MemSpace {
    match ss {
        Some(LdSs::Global) => MemSpace::Global,
        Some(LdSs::Shared | LdSs::SharedCta | LdSs::SharedCluster) => MemSpace::Shared,
        _ => MemSpace::Other,
    }
}

fn st_ss(ss: &Option<StSs>) -> MemSpace {
    match ss {
        Some(StSs::Global) => MemSpace::Global,
        Some(StSs::Shared | StSs::SharedCta | StSs::SharedCluster) => MemSpace::Shared,
        _ => MemSpace::Other,
    }
}

fn atom_space(space: &Option<AtomSpace>) -> MemSpace {
    match space {
        Some(AtomSpace::Global) => MemSpace::Global,
        Some(AtomSpace::Shared | AtomSpace::SharedCta | AtomSpace::SharedCluster) => MemSpace::Shared,
        _ => MemSpace::Other,
    }
}

fn red0_space(space: &Option<RedSpace0>) -> MemSpace {
    match space {
        Some(RedSpace0::Global) => MemSpace::Global,
        Some(RedSpace0::Shared | RedSpace0::SharedCta | RedSpace0::SharedCluster) => MemSpace::Shared,
        _ => MemSpace::Other,
    }
}

fn red1_space(space: &Option<RedSpace1>) -> MemSpace {
    match space {
        Some(RedSpace1::Global) => MemSpace::Global,
        Some(RedSpace1::Shared | RedSpace1::SharedCta | RedSpace1::SharedCluster) => MemSpace::Shared,
        _ => MemSpace::Other,
    }
}

fn prefetch_space(space: &Option<PrefetchSpace>) -> MemSpace {
    match space {
        Some(PrefetchSpace::Global) => MemSpace::Global,
        _ => MemSpace::Other,
    }
}

/// Extract the memory state space for an instruction.
/// Returns `None` if `inst` is not a memory instruction.
fn inst_memory_space(inst: &Inst) -> Option<MemSpace> {
    use Inst::*;
    Some(match inst {
        // ld (ss-qualified)
        LdWeakSsCopLevelCacheHintLevelPrefetchSizeVecType(x) => ld_ss(&x.ss),
        LdWeakSsLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintLevelPrefetchSizeVecType(x) => ld_ss(&x.ss),
        LdVolatileSsLevelPrefetchSizeVecType(x) => ld_ss(&x.ss),
        LdRelaxedScopeSsLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintLevelPrefetchSizeVecType(x) => ld_ss(&x.ss),
        LdAcquireScopeSsLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintLevelPrefetchSizeVecType(x) => ld_ss(&x.ss),
        // ld.global (always global, including __ldg / non-coherent variants)
        LdGlobalCopNcLevelCacheHintLevelPrefetchSizeType(_)
        | LdGlobalCopNcLevelCacheHintLevelPrefetchSizeVecType(_)
        | LdGlobalNcLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintLevelPrefetchSizeType(_)
        | LdGlobalNcLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintLevelPrefetchSizeVecType(_)
        | LdMmioRelaxedSysGlobalType(_) => MemSpace::Global,
        // ldu — .global only
        LduSsType(_) | LduSsVecType(_) => MemSpace::Global,
        // ldmatrix — always .shared
        LdmatrixSyncAlignedShapeNumTransSsType(_)
        | LdmatrixSyncAlignedM8n16NumSsDstFmtSrcFmt(_)
        | LdmatrixSyncAlignedM16n16NumTransSsDstFmtSrcFmt(_) => MemSpace::Shared,
        // st (ss-qualified)
        StWeakSsCopLevelCacheHintVecType(x) => st_ss(&x.ss),
        StWeakSsLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintVecType(x) => st_ss(&x.ss),
        StVolatileSsVecType(x) => st_ss(&x.ss),
        StRelaxedScopeSsLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintVecType(x) => st_ss(&x.ss),
        StReleaseScopeSsLevel1EvictionPriorityLevel2EvictionPriorityLevelCacheHintVecType(x) => st_ss(&x.ss),
        StMmioRelaxedSysGlobalType(_) => MemSpace::Global,
        // st.async: section_0 uses SharedCluster, section_1 uses Global
        StAsyncSemScopeSsCompletionMechanismVecType(_) => MemSpace::Shared,
        StAsyncMmioSemScopeSsType(_) => MemSpace::Global,
        // st.bulk — always .shared::cta
        StBulkWeakSharedCta(_) => MemSpace::Shared,
        // atom (space-qualified)
        AtomSemScopeSpaceOpLevelCacheHintType(x) => atom_space(&x.space),
        AtomSemScopeSpaceOpType(x) => atom_space(&x.space),
        AtomSemScopeSpaceCasB16(x) => atom_space(&x.space),
        AtomSemScopeSpaceCasB128(x) => atom_space(&x.space),
        AtomSemScopeSpaceExchLevelCacheHintB128(x) => atom_space(&x.space),
        AtomSemScopeSpaceAddNoftzLevelCacheHintF16(x) => atom_space(&x.space),
        AtomSemScopeSpaceAddNoftzLevelCacheHintF16x2(x) => atom_space(&x.space),
        AtomSemScopeSpaceAddNoftzLevelCacheHintBf16(x) => atom_space(&x.space),
        AtomSemScopeSpaceAddNoftzLevelCacheHintBf16x2(x) => atom_space(&x.space),
        // atom.global (always global)
        AtomSemScopeGlobalAddLevelCacheHintVec32BitF32(_)
        | AtomSemScopeGlobalOpNoftzLevelCacheHintVec16BitHalfWordType(_)
        | AtomSemScopeGlobalOpNoftzLevelCacheHintVec32BitPackedType(_) => MemSpace::Global,
        // red (space-qualified, section_0 and section_1)
        RedOpSpaceSemScopeLevelCacheHintType(x) => red0_space(&x.space),
        RedAddSpaceSemScopeNoftzLevelCacheHintF16(x) => red0_space(&x.space),
        RedAddSpaceSemScopeNoftzLevelCacheHintF16x2(x) => red0_space(&x.space),
        RedAddSpaceSemScopeNoftzLevelCacheHintBf16(x) => red0_space(&x.space),
        RedAddSpaceSemScopeNoftzLevelCacheHintBf16x2(x) => red0_space(&x.space),
        RedAddSpaceSemScopeLevelCacheHintVec32BitF32(x) => red1_space(&x.space),
        RedOpSpaceSemScopeNoftzLevelCacheHintVec16BitHalfWordType(x) => red1_space(&x.space),
        RedOpSpaceSemScopeNoftzLevelCacheHintVec32BitPackedType(x) => red1_space(&x.space),
        // red.async — sections 0–3 are .shared::cluster, section 4 is .global
        RedAsyncSemScopeSsCompletionMechanismOpType(_)
        | RedAsyncSemScopeSsCompletionMechanismOpType1(_)
        | RedAsyncSemScopeSsCompletionMechanismOpType2(_)
        | RedAsyncSemScopeSsCompletionMechanismAddType(_) => MemSpace::Shared,
        RedAsyncMmioSemScopeSsAddType(_) => MemSpace::Global,
        // prefetch
        PrefetchSpaceLevel(x) => prefetch_space(&x.space),
        PrefetchGlobalLevelEvictionPriority(_) => MemSpace::Global,
        PrefetchuL1(_) | PrefetchTensormapSpaceTensormap(_) => MemSpace::Other,
        // Not memory instructions
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// PTX special registers that encode the thread/block index.
/// A register whose value ultimately comes from one of these is "tid-dependent".
/// These come from the DDG token stream, so they remain string-based.
fn is_tid_special_register(reg: &str) -> bool {
    matches!(
        reg,
        "%tid.x" | "%tid.y" | "%tid.z"
            | "%ctaid.x" | "%ctaid.y" | "%ctaid.z"
            | "%ntid.x" | "%ntid.y" | "%ntid.z"
            | "%nctaid.x" | "%nctaid.y" | "%nctaid.z"
            | "%laneid" | "%warpid"
    )
}

/// How many loops contain `block_id` in their body.
fn loop_depth(block_id: BlockId, cfg: &ControlFlowGraph) -> usize {
    cfg.loops.iter().filter(|l| l.body.contains(&block_id)).count()
}

// ---------------------------------------------------------------------------
// Address pattern classification
// ---------------------------------------------------------------------------

/// Classify the address access pattern for a single memory instruction.
///
/// Strategy: BFS backwards through the DDG from the instruction's use-set.
/// Collect all "root" registers — those with no DDG predecessors.
///
/// Classify based on what roots we find:
///   - Any root that IS a tid/ctaid special register → TidLinear
///   - Any root that is produced by a `ld.*` instruction → Indirect
///   - No predecessors at all on the immediate use-set → Invariant
///   - Otherwise, if inside a loop → LoopVariable; else → Complex
fn classify_address_pattern(
    mem_id: InstrId,
    ddg: &DataDependencyGraph,
    cfg: &ControlFlowGraph,
) -> AddressPattern {
    // Build reverse edge map: consumer → producers
    let mut predecessors: HashMap<InstrId, Vec<InstrId>> = HashMap::new();
    for edge in &ddg.edges {
        predecessors.entry(edge.to).or_default().push(edge.from);
    }

    let uses = match ddg.uses.get(&mem_id) {
        Some(u) => u.clone(),
        None => return AddressPattern::Invariant,
    };

    if uses.is_empty() {
        return AddressPattern::Invariant;
    }

    let mut visited: HashSet<InstrId> = HashSet::new();
    let mut queue: Vec<InstrId> = vec![mem_id];
    let mut found_tid = false;
    let mut found_indirect = false;
    let mut reachable_count = 0usize;

    while let Some(cur) = queue.pop() {
        if !visited.insert(cur) {
            continue;
        }
        reachable_count += 1;

        if let Some(cur_uses) = ddg.uses.get(&cur) {
            for u in cur_uses {
                if is_tid_special_register(u) {
                    found_tid = true;
                }
            }
        }

        // DDG nodes store PTX text strings — check if a predecessor is a global load.
        // This is intentionally string-based because the DDG itself is string-based.
        if cur != mem_id {
            if let Some(node_text) = ddg.nodes.get(&cur) {
                let opcode_part = node_text
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_start_matches('@')
                    .trim_start_matches('!');
                let clean = if opcode_part.starts_with('%') {
                    node_text.split_whitespace().nth(1).unwrap_or("")
                } else {
                    opcode_part
                };
                if clean.starts_with("ld.global") || clean.starts_with("ld.nc") {
                    found_indirect = true;
                }
            }
        }

        if let Some(preds) = predecessors.get(&cur) {
            for &p in preds {
                if !visited.contains(&p) {
                    queue.push(p);
                }
            }
        }
    }

    if found_indirect {
        return AddressPattern::Indirect;
    }
    if found_tid {
        return AddressPattern::TidLinear;
    }
    if reachable_count <= 1 {
        return AddressPattern::Invariant;
    }

    if loop_depth(mem_id.0, cfg) > 0 {
        AddressPattern::LoopVariable
    } else {
        AddressPattern::Complex
    }
}

// ---------------------------------------------------------------------------
// Stall / miss rate model
// ---------------------------------------------------------------------------

fn stall_model(pattern: &AddressPattern, depth: usize, is_shared: bool) -> (f64, f64) {
    if is_shared {
        return (0.02, 20.0);
    }

    match pattern {
        AddressPattern::TidLinear => match depth {
            0 => (0.05, 20.0),
            1 => (0.20, 40.0),
            _ => (0.45, 100.0),
        },
        AddressPattern::Invariant => (0.03, 5.0),
        AddressPattern::LoopVariable => match depth {
            0 | 1 => (0.40, 80.0),
            2 => (0.70, 150.0),
            _ => (0.90, 200.0),
        },
        AddressPattern::Indirect => (0.92, 200.0),
        AddressPattern::Complex => match depth {
            0 => (0.30, 60.0),
            _ => (0.60, 120.0),
        },
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Build a `MemoryProfile` for every memory instruction occurrence in the CFG.
pub fn build_memory_profiles(
    cfg: &ControlFlowGraph,
    ddg: &DataDependencyGraph,
    occurrence_counts: &HashMap<InstrId, u64>,
) -> Vec<MemoryProfile> {
    let mut profiles: Vec<MemoryProfile> = Vec::new();

    for block in &cfg.blocks {
        let mut instr_idx = 0usize;
        for stmt in &block.statements {
            let inst = match stmt {
                FunctionStatement::Instruction { instruction, .. } => &instruction.inst,
                _ => continue,
            };

            let id: InstrId = (block.id, instr_idx);
            instr_idx += 1;

            let space = match inst_memory_space(inst) {
                Some(s) => s,
                None => continue,
            };

            let opcode = statement_opcode(stmt).unwrap_or_default();
            let occurrences = *occurrence_counts.get(&id).unwrap_or(&0);
            let depth = loop_depth(block.id, cfg);
            let is_shared = space == MemSpace::Shared;

            let pattern = classify_address_pattern(id, ddg, cfg);
            let (miss_rate, stall_cycles) = stall_model(&pattern, depth, is_shared);
            let total_stall = stall_cycles * occurrences as f64;

            profiles.push(MemoryProfile {
                block_id: block.id,
                instr_idx: id.1,
                opcode,
                occurrences,
                loop_depth: depth,
                address_pattern: pattern,
                estimated_miss_rate: miss_rate,
                estimated_stall_cycles: stall_cycles,
                estimated_total_stall_cycles: total_stall,
            });
        }
    }

    profiles
}
