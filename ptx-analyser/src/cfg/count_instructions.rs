<<<<<<< Updated upstream
use ptx_parser::r#type::FunctionStatement;
use crate::{Dim3, Parameter};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use super::common::{BlockId, ControlFlowGraph, statement_opcode};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstructionAnalysisReport {
    pub kernel_name: Option<String>,
    pub power_consumption_joules: Option<f64>,
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub parameters: Vec<Parameter>,
    pub total_instructions: u64,
    pub instruction_occurrences: BTreeMap<String, u64>,
}

/// Walk the CFG and return a map of opcode weighted execution count.
/// Loop bodies are multiplied by their `min_iters` from `cfg.loops`.
#[allow(dead_code)]
pub fn collect_instruction_counts(cfg: &ControlFlowGraph) -> HashMap<String, u64> {
    let mut visited = BTreeSet::new();
    let mut total = 0u64;
    let mut scope_instructions = 0u64;
    let mut scope_iterations = 1u64;
    let mut occurrences = HashMap::new();
    count_instructions_recursive(
        cfg.entry,
        cfg,
        &mut visited,
        &mut total,
        &mut scope_instructions,
        &mut scope_iterations,
        &mut occurrences,
    );
    occurrences
=======
use ptx_parser::r#type::{FunctionStatement, MetaTag, Operand};
use std::collections::{BTreeSet, HashMap};
use crate::{Parameter, cfg::common::statement_opcode};

use super::common::{BasicBlock, BlockId, ControlFlowGraph};


#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct Bound {
    pub min: i128,
    pub max: i128,
>>>>>>> Stashed changes
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
                            iterations_count = operand.value.parse().unwrap();
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


/// Recursively count all instructions in the CFG starting from a given block.
fn count_instructions_recursive(
    block_id: BlockId,
    cfg: &ControlFlowGraph,
    visited: &mut BTreeSet<BlockId>,
    total_instructions: &mut u64,
    scope_instructions: &mut u64,
    scope_iterations: &mut u64,
    registers: &HashMap<String, u64>,
    instruction_occurrences: &mut HashMap<String, u64>,
    last_meta_loop: &mut Option<MetaTag>,
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

    // Count instructions in current block1x
    let block = &cfg.blocks[block_id];
    let block_predecessor_count = cfg.predecessors.get(&block_id).unwrap().len();

    // Debug: Print if the block has no instructions, this should be impossible.
    if block.statements.is_empty() {
        println!("[ERROR] Block {} has no statements", block_id);
    }

    println!(
        "[VISIT] Visiting block {}, current scope instructions: {}, current scope iterations: {}",
        block_id, *scope_instructions, *scope_iterations
    );

    // Itterate over all instructions in the block to find any setp instructions that might affect loop iteration counts
    for stmt in &block.statements {
        if let FunctionStatement::Instruction { instruction, .. } = stmt {
            let formatted_inst = statement_opcode(&stmt).unwrap_or("Unknown".to_string());

            println!(
                "[INSTRUCTION] Found instruction in block {}: {:?} {:?}",
                block_id, formatted_inst, statement_opcode(&stmt)
            );

            *scope_instructions += 1;
            instruction_occurrences.entry(formatted_inst)
                .and_modify(|count| *count += *scope_iterations)
                .or_insert(*scope_iterations);
        } else if let FunctionStatement::Meta { directive, .. } = stmt {
            if let MetaTag::Loop { .. } = directive.tag {
                *last_meta_loop = Some(directive.tag.clone());
            }
            // Handle meta directives if needed for power estimation (e.g., loop annotations)
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
                visited,
                total_instructions,
                scope_instructions,
                scope_iterations,
                registers,
                instruction_occurrences,
                last_meta_loop,
            );
        } else if successors.len() > 1 {
            // Check if last_meta_loop is Some, and if so read the loop bounds
            let mut branch_iterations_count: u64 = 1;
            if let Some(MetaTag::Loop { max_iters, min_iters, .. }) = last_meta_loop {
                println!(
                    "[META_LOOP] Found loop meta directive in block {}: min_iters={}, max_iters={}",
                    block_id, min_iters, max_iters
                );

                branch_iterations_count = min_iters.clone().into();
                last_meta_loop.take(); // Clear the last_meta_loop after using it
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
                    visited,
                    total_instructions,
                    &mut block_b_scope_instructions,
                    &mut block_b_scope_iterations,
                    registers,
                    instruction_occurrences,
                    last_meta_loop,
                );

                println!(
                    "[A_TRUE_BRANCH] Visiting true branch (block {}) with scope iterations {} from block {}",
                    successor_a_id, block_a_scope_iterations, block_id
                );

                count_instructions_recursive(
                    successor_a_id,
                    cfg,
                    visited,
                    total_instructions,
                    &mut block_a_scope_instructions,
                    &mut block_a_scope_iterations,
                    registers,
                    instruction_occurrences,
                    last_meta_loop,
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
                    visited,
                    total_instructions,
                    &mut block_a_scope_instructions,
                    &mut block_a_scope_iterations,
                    registers,
                    instruction_occurrences,
                    last_meta_loop,
                );

                println!(
                    "[A_FALSE_BRANCH] Visiting true branch (block {}) with scope iterations {} from block {}",
                    successor_b_id, block_b_scope_iterations, block_id
                );

                count_instructions_recursive(
                    successor_b_id,
                    cfg,
                    visited,
                    total_instructions,
                    &mut block_b_scope_instructions,
                    &mut block_b_scope_iterations,
                    registers,
                    instruction_occurrences,
                    last_meta_loop,
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

    // Count total instructions by recursively walking the CFG tree
    let mut visited = BTreeSet::new();
    let mut total_instructions: u64 = 0;
    let mut scope_instructions: u64 = 0;
    let mut scope_iterations: u64 = (grid_x * grid_y * grid_z * block_x * block_y * block_z).into();

    // Registers for tracking loop iteration counts (e.g., from setp instructions)
    let registers: HashMap<String, u64> = HashMap::new();

    // Track occurrences of each instruction type for potential use in a more detailed power model
    let mut instruction_occurrences: HashMap<String, u64> = HashMap::new();

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
        &cfg,
        &mut visited,
        &mut total_instructions,
        &mut scope_instructions,
        &mut scope_iterations,
        &registers,
        &mut instruction_occurrences,
        &mut None,  
    );
<<<<<<< Updated upstream

    let report = InstructionAnalysisReport {
        kernel_name: kernel_name.clone(),
        power_consumption_joules: *power_consumption_joules,
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
    };

    let json = serde_json::to_string_pretty(&report)
        .expect("failed to serialize instruction analysis report to JSON");

    if let Some(output_path) = output_json_path {
        fs::write(output_path, json)
            .unwrap_or_else(|e| panic!("failed to write JSON report to {}: {}", output_path.display(), e));
        println!("[Output] JSON report written to {}", output_path.display());
    } else {
        println!("{json}");
=======
// 
    println!("[Output] Total instructions for the CFG: {}", total_instructions);
    println!("[Output] CSV Rows:");
    for (inst, count) in &instruction_occurrences {
        println!("{}, {}, {}, {}, {}", csv_kernel_name.as_ref().unwrap_or(&"Unknown".into()), inst, count, total_instructions, csv_power_consumption_joules.unwrap_or(0.0));
>>>>>>> Stashed changes
    }
}
