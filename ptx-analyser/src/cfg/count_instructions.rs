use ptx_parser::r#type::{FunctionStatement, MetaTag, Operand};
use crate::{Dim3, Parameter};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use super::common::{BasicBlock, BlockId, ControlFlowGraph, statement_opcode};
use super::ddg::{build_ddg, InstrId};
use super::mem_analysis::{build_memory_profiles, MemoryProfile};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstructionAnalysisReport {
    pub kernel_name: Option<String>,
    pub power_consumption_joules: Option<f64>,
    pub kernel_duration_s: Option<f64>,
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub parameters: Vec<Parameter>,
    pub total_instructions: u64,
    pub instruction_occurrences: BTreeMap<String, u64>,
    /// Consecutive opcode pairs where the first instruction produces a value the
    /// second consumes (RAW dependency confirmed by DDG). Key: "opA,opB".
    pub dependent_pairs: BTreeMap<String, u64>,
    /// Consecutive opcode pairs with no RAW dependency between them.
    /// Key: "opA,opB".
    pub independent_pairs: BTreeMap<String, u64>,
    /// Per-occurrence memory access profiles for every ld/st/atom instruction.
    /// Each entry is a distinct CFG site (block_id, instr_idx), so two ld.global.f32
    /// instructions in different blocks get separate profiles with different
    /// address patterns, loop depths, and estimated cache behaviour.
    pub memory_profiles: Vec<MemoryProfile>,
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct Bound {
    pub min: i128,
    pub max: i128,
}

#[allow(dead_code)]
pub trait HasLoopBounds {
    fn loop_bounds(&self) -> Bound;
}

impl HasLoopBounds for ptx_parser::r#type::instruction::ld::section_0::Type {
    fn loop_bounds(&self) -> Bound {
        match self {
            ptx_parser::r#type::instruction::ld::section_0::Type::U8 => Bound {
                min: 0,
                max: u8::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::U16 => Bound {
                min: 0,
                max: u16::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::U32 => Bound {
                min: 0,
                max: u32::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::U64 => Bound {
                min: 0,
                max: u64::MAX as i128,
            },

            ptx_parser::r#type::instruction::ld::section_0::Type::S8 => Bound {
                min: i8::MIN as i128,
                max: i8::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::S16 => Bound {
                min: i16::MIN as i128,
                max: i16::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::S32 => Bound {
                min: i32::MIN as i128,
                max: i32::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::S64 => Bound {
                min: i64::MIN as i128,
                max: i64::MAX as i128,
            },

            ptx_parser::r#type::instruction::ld::section_0::Type::B8 => Bound {
                min: 0,
                max: u8::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::B16 => Bound {
                min: 0,
                max: u16::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::B32 => Bound {
                min: 0,
                max: u32::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::B64 => Bound {
                min: 0,
                max: u64::MAX as i128,
            },
            ptx_parser::r#type::instruction::ld::section_0::Type::B128 => Bound {
                min: 0,
                max: i128::MAX,
            },

            // floats handled separately if needed
            ptx_parser::r#type::instruction::ld::section_0::Type::F32 => Bound {
                min: i128::MIN,
                max: i128::MAX,
            }, // placeholder
            ptx_parser::r#type::instruction::ld::section_0::Type::F64 => Bound {
                min: i128::MIN,
                max: i128::MAX,
            },
        }
    }
}

pub(crate) struct cmp_info {
    target_label: String,
    iterations_count: u64,
}

pub trait GetCmpInfo {
    fn get_cmp_info(&self, registers: &HashMap<String, u64>) -> Option<cmp_info>;
}

impl GetCmpInfo for BasicBlock {
    fn get_cmp_info(&self, registers: &HashMap<String, u64>) -> Option<cmp_info> {
        let mut is_branch = false;
        let mut target_label: String = String::new();
        let mut iterations_count: u64 = 1;

        for stmt in &self.statements {
            if let FunctionStatement::Instruction { instruction, .. } = stmt {
                if let ptx_parser::r#type::instruction::Inst::SetpCmpopFtzType(inst) =
                    &instruction.inst
                {
                    if let ptx_parser::r#type::GeneralOperand::Single { operand, .. } = &inst.b {
                        print!(
                            "[DEBUG] Found SetpCmpopFtzType with operand: {:?}\n",
                            operand
                        );
                        if let Operand::Register { operand, span } = operand {
                            iterations_count = *registers.get(&operand.name).unwrap_or(&1);
                        }
                        if let Operand::Immediate { operand, .. } = operand {
                            if let Some(hex) = operand.value.strip_prefix("0d") {
                                if let Ok(bits) = u64::from_str_radix(hex, 16) {
                                    iterations_count = f64::from_bits(bits) as u64;
                                }
                            } else if let Some(hex) = operand.value.strip_prefix("0f") {
                                if let Ok(bits) = u32::from_str_radix(hex, 16) {
                                    iterations_count = f32::from_bits(bits) as u64;
                                }
                            } else if let Ok(val) = operand.value.parse() {
                                iterations_count = val;
                            }
                        }
                    }
                }
                if let ptx_parser::r#type::instruction::Inst::BraUni(inst) = &instruction.inst {
                    is_branch = true;
                    if let ptx_parser::r#type::GeneralOperand::Single { operand, .. } = &inst.tgt {
                        if let Operand::Symbol { name, .. } = operand {
                            target_label = name.clone();
                        }
                    }
                }
            }
        }

        if is_branch {
            return Some(cmp_info {
                target_label,
                iterations_count,
            });
        }

        None
    }
}

/// State carried forward across the instruction walk to emit consecutive pairs.
struct PrevInstr {
    opcode: String,
    id: InstrId,
}

/// Recursively count all instructions in the CFG starting from a given block.
fn count_instructions_recursive(
    block_id: BlockId,
    cfg: &ControlFlowGraph,
    ddg_edges: &HashSet<(InstrId, InstrId)>,
    visited: &mut BTreeSet<BlockId>,
    total_instructions: &mut u64,
    scope_instructions: &mut u64,
    scope_iterations: &mut u64,
    registers: &HashMap<String, u64>,
    instruction_occurrences: &mut HashMap<String, u64>,
    per_instr_occurrences: &mut HashMap<InstrId, u64>,
    dependent_pairs: &mut HashMap<String, u64>,
    independent_pairs: &mut HashMap<String, u64>,
    last_meta_loop: &mut Option<MetaTag>,
    prev: &mut Option<PrevInstr>,
) {
    // Avoid infinite loops by tracking visited blocks
    if visited.contains(&block_id) || cfg.successors.get(&block_id).is_none() {
        *total_instructions += *scope_instructions * *scope_iterations;
        println!(
            "[RETURN_END] Incrementing total instructions by {}, returning from block {}",
            *scope_instructions * *scope_iterations,
            block_id
        );
        return;
    }

    visited.insert(block_id);

    let block = &cfg.blocks[block_id];
    let _block_predecessor_count = cfg.predecessors.get(&block_id).unwrap().len();

    if block.statements.is_empty() {
        println!("[ERROR] Block {} has no statements", block_id);
    }

    println!(
        "[VISIT] Visiting block {}, current scope instructions: {}, current scope iterations: {}",
        block_id, *scope_instructions, *scope_iterations
    );

    let mut instr_idx = 0usize;
    for stmt in &block.statements {
        if let FunctionStatement::Instruction { .. } = stmt {
            let cur_id: InstrId = (block_id, instr_idx);
            instr_idx += 1;

            let cur_opcode = statement_opcode(stmt).unwrap_or("Unknown".to_string());

            println!(
                "[INSTRUCTION] Found instruction in block {}: {:?}",
                block_id, cur_opcode
            );

            // Emit pair weighted by scope_iterations, classified via DDG
            if let Some(p) = prev.as_ref() {
                let pair_key = format!("{},{}", p.opcode, cur_opcode);
                let target = if ddg_edges.contains(&(p.id, cur_id)) {
                    &mut *dependent_pairs
                } else {
                    &mut *independent_pairs
                };
                target
                    .entry(pair_key)
                    .and_modify(|c| *c += *scope_iterations)
                    .or_insert(*scope_iterations);
            }
            *prev = Some(PrevInstr { opcode: cur_opcode.clone(), id: cur_id });

            *scope_instructions += 1;
            instruction_occurrences
                .entry(cur_opcode)
                .and_modify(|count| *count += *scope_iterations)
                .or_insert(*scope_iterations);
            per_instr_occurrences
                .entry(cur_id)
                .and_modify(|c| *c += *scope_iterations)
                .or_insert(*scope_iterations);
        } else if let FunctionStatement::Meta { directive, .. } = stmt {
            if let MetaTag::Loop { .. } = directive.tag {
                *last_meta_loop = Some(directive.tag.clone());
            }
            println!(
                "[META] Found meta directive in block {}: {:?}",
                block_id, directive
            );
        }
    }

    // Recurse into successors
    if let Some(successors) = cfg.successors.get(&block_id) {
        if successors.is_empty() {
            println!("[NO_SUCCESSORS] Block {} has no successors", block_id);
        } else if successors.len() == 1 {
            println!(
                "[SINGLE_SUCCESSOR] Block {} has a single successor {}, continuing with same scope iterations {}",
                block_id,
                *successors.iter().nth(0).unwrap(),
                *scope_iterations
            );
            count_instructions_recursive(
                *successors.iter().nth(0).unwrap(),
                cfg,
                ddg_edges,
                visited,
                total_instructions,
                scope_instructions,
                scope_iterations,
                registers,
                instruction_occurrences,
                per_instr_occurrences,
                dependent_pairs,
                independent_pairs,
                last_meta_loop,
                prev,
            );
        } else if successors.len() > 1 {
            let mut branch_iterations_count: u64 = 1;
            if let Some(MetaTag::Loop { max_iters, min_iters, .. }) = last_meta_loop {
                println!(
                    "[META_LOOP] Found loop meta directive in block {}: min_iters={}, max_iters={}",
                    block_id, min_iters, max_iters
                );
                branch_iterations_count = min_iters.clone().into();

                if branch_iterations_count == 0 {
                    println!(
                        "[META_LOOP_WARNING] Loop in block {} has zero iterations according to meta directive, defaulting to 1",
                        block_id
                    );
                }

                last_meta_loop.take();
            }

            let branch_info = block.get_cmp_info(registers).unwrap();
            let is_loop = cfg.loops.iter().any(|loop_info| loop_info.header == block_id);

            let successor_a_id = *successors.iter().nth(0).unwrap();
            let successor_b_id = *successors.iter().nth(1).unwrap();

            let successor_a_block = &cfg.blocks[successor_a_id];
            let is_block_a_true_branch = successor_a_block.label.is_some()
                && successor_a_block.label.as_ref().unwrap() == &branch_info.target_label;

            if is_loop {
                println!(
                    "[LOOP_BRANCH] Loop branch found at block {}: true branch is {}, iteration count is {}",
                    block_id,
                    if is_block_a_true_branch { "A" } else { "B" },
                    branch_iterations_count
                );
            } else {
                println!(
                    "[BRANCH] Branch found at block {}: true branch is {}, iteration count is {}",
                    block_id,
                    if is_block_a_true_branch { "A" } else { "B" },
                    branch_iterations_count
                );
            }

            let mut block_a_scope_instructions = 0;
            let mut block_b_scope_instructions = 0;
            // Each branch gets its own prev state so cross-branch pairs don't bleed over.
            let mut branch_a_prev = prev.as_ref().map(|p| PrevInstr { opcode: p.opcode.clone(), id: p.id });
            let mut branch_b_prev = prev.as_ref().map(|p| PrevInstr { opcode: p.opcode.clone(), id: p.id });

            if is_block_a_true_branch {
                let mut block_a_scope_iterations = *scope_iterations;
                let mut block_b_scope_iterations = if is_loop {
                    *scope_iterations * branch_iterations_count
                } else {
                    *scope_iterations
                };

                println!(
                    "[A_TRUE_BRANCH] Visiting false branch (block {}) with scope iterations {} from block {}",
                    successor_b_id, block_b_scope_iterations, block_id
                );
                count_instructions_recursive(
                    successor_b_id,
                    cfg,
                    ddg_edges,
                    visited,
                    total_instructions,
                    &mut block_b_scope_instructions,
                    &mut block_b_scope_iterations,
                    registers,
                    instruction_occurrences,
                    per_instr_occurrences,
                    dependent_pairs,
                    independent_pairs,
                    last_meta_loop,
                    &mut branch_b_prev,
                );

                println!(
                    "[A_TRUE_BRANCH] Visiting true branch (block {}) with scope iterations {} from block {}",
                    successor_a_id, block_a_scope_iterations, block_id
                );
                count_instructions_recursive(
                    successor_a_id,
                    cfg,
                    ddg_edges,
                    visited,
                    total_instructions,
                    &mut block_a_scope_instructions,
                    &mut block_a_scope_iterations,
                    registers,
                    instruction_occurrences,
                    per_instr_occurrences,
                    dependent_pairs,
                    independent_pairs,
                    last_meta_loop,
                    &mut branch_a_prev,
                );
            } else {
                let mut block_a_scope_iterations = if is_loop {
                    *scope_iterations * branch_iterations_count
                } else {
                    *scope_iterations
                };
                let mut block_b_scope_iterations = *scope_iterations;

                println!(
                    "[A_FALSE_BRANCH] Visiting false branch (block {}) with scope iterations {} from block {}",
                    successor_a_id, block_a_scope_iterations, block_id
                );
                count_instructions_recursive(
                    successor_a_id,
                    cfg,
                    ddg_edges,
                    visited,
                    total_instructions,
                    &mut block_a_scope_instructions,
                    &mut block_a_scope_iterations,
                    registers,
                    instruction_occurrences,
                    per_instr_occurrences,
                    dependent_pairs,
                    independent_pairs,
                    last_meta_loop,
                    &mut branch_a_prev,
                );

                println!(
                    "[A_FALSE_BRANCH] Visiting true branch (block {}) with scope iterations {} from block {}",
                    successor_b_id, block_b_scope_iterations, block_id
                );
                count_instructions_recursive(
                    successor_b_id,
                    cfg,
                    ddg_edges,
                    visited,
                    total_instructions,
                    &mut block_b_scope_instructions,
                    &mut block_b_scope_iterations,
                    registers,
                    instruction_occurrences,
                    per_instr_occurrences,
                    dependent_pairs,
                    independent_pairs,
                    last_meta_loop,
                    &mut branch_b_prev,
                );
            }

            *total_instructions += *scope_instructions * *scope_iterations;
            println!(
                "[RETURN_SCOPE] Incrementing total instructions by {}, returning from block {}",
                *scope_instructions * *scope_iterations,
                block_id
            );
        }
    }
}

// Analyze the cfg to get a power consumption estimate
pub fn analyze_cfg(
    cfg: &ControlFlowGraph,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    params: &Vec<Parameter>,
    output_json_path: &Option<PathBuf>,
    kernel_name: &Option<String>,
    power_consumption_joules: &Option<f64>,
    kernel_duration_s: &Option<f64>,
) {
    println!(
        "[ANALYZE_CFGS] Grid dimensions: ({}, {}, {}), Block dimensions: ({}, {}, {})",
        grid_x, grid_y, grid_z, block_x, block_y, block_z
    );
    if !params.is_empty() {
        println!("[ANALYZE_CFGS] Kernel parameters:");
        for param in params {
            println!(
                "  - {} ({:?}, {} bytes)",
                param.name, param.r#type, param.size
            );
        }
    }

    // Build DDG and flatten edges into a set for O(1) dependency lookup.
    let ddg = build_ddg(cfg);
    let ddg_edges: HashSet<(InstrId, InstrId)> = ddg.edges.iter().map(|e| (e.from, e.to)).collect();

    let mut visited = BTreeSet::new();
    let mut total_instructions: u64 = 0;
    let mut scope_instructions: u64 = 0;
    let mut scope_iterations: u64 = 1;

    let registers: HashMap<String, u64> = HashMap::new();
    let mut instruction_occurrences: HashMap<String, u64> = HashMap::new();
    let mut dependent_pairs: HashMap<String, u64> = HashMap::new();
    let mut independent_pairs: HashMap<String, u64> = HashMap::new();
    let mut per_instr_occurrences: HashMap<InstrId, u64> = HashMap::new();
    let mut prev: Option<PrevInstr> = None;

    if let Some(block) = cfg.blocks.get(cfg.entry) {
        println!(
            "[ANALYZE_CFGS] Starting analysis from entry block {} with label {:?} and {} successors",
            block.id,
            block.label,
            cfg.successors.get(&block.id).map_or(0, |succs| succs.len())
        );
    }

    count_instructions_recursive(
        cfg.entry,
        cfg,
        &ddg_edges,
        &mut visited,
        &mut total_instructions,
        &mut scope_instructions,
        &mut scope_iterations,
        &registers,
        &mut instruction_occurrences,
        &mut per_instr_occurrences,
        &mut dependent_pairs,
        &mut independent_pairs,
        &mut None,
        &mut prev,
    );

    let memory_profiles = build_memory_profiles(cfg, &ddg, &per_instr_occurrences);

    let report = InstructionAnalysisReport {
        kernel_name: kernel_name.clone(),
        power_consumption_joules: *power_consumption_joules,
        kernel_duration_s: *kernel_duration_s,
        grid_dim: Dim3 {
            x: grid_x,
            y: grid_y,
            z: grid_z,
        },
        block_dim: Dim3 {
            x: block_x,
            y: block_y,
            z: block_z,
        },
        parameters: params.clone(),
        total_instructions,
        instruction_occurrences: instruction_occurrences
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        dependent_pairs: dependent_pairs
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        independent_pairs: independent_pairs
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        memory_profiles,
    };

    let json = serde_json::to_string_pretty(&report)
        .expect("failed to serialize instruction analysis report to JSON");

    if let Some(output_path) = output_json_path {
        fs::write(output_path, json)
            .unwrap_or_else(|e| panic!("failed to write JSON report to {}: {}", output_path.display(), e));
        println!("[Output] JSON report written to {}", output_path.display());
    } else {
        println!("{json}");
    }
}
