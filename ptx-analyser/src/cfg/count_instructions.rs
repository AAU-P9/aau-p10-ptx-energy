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
}

/// Recursively count all instructions in the CFG starting from a given block.
///
/// Loop iteration counts and body membership are read directly from
/// `cfg.loops`, which is populated during CFG construction from `@META Loop`
/// annotations.
pub fn count_instructions_recursive(
    block_id: BlockId,
    cfg: &ControlFlowGraph,
    visited: &mut BTreeSet<BlockId>,
    total_instructions: &mut u64,
    scope_instructions: &mut u64,
    scope_iterations: &mut u64,
    instruction_occurrences: &mut HashMap<String, u64>,
) {
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

    if block.statements.is_empty() {
        println!("[ERROR] Block {} has no statements", block_id);
    }

    println!(
        "[VISIT] Visiting block {}, current scope instructions: {}, current scope iterations: {}",
        block_id, *scope_instructions, *scope_iterations
    );

    for stmt in &block.statements {
        if let FunctionStatement::Instruction { .. } = stmt {
            let formatted_inst = statement_opcode(stmt).unwrap_or("Unknown".to_string());

            println!(
                "[INSTRUCTION] Found instruction in block {}: {:?}",
                block_id, formatted_inst
            );

            *scope_instructions += 1;
            instruction_occurrences
                .entry(formatted_inst)
                .and_modify(|count| *count += *scope_iterations)
                .or_insert(*scope_iterations);
        }
    }

    if let Some(successors) = cfg.successors.get(&block_id) {
        if successors.is_empty() {
            println!("[NO_SUCCESSORS] Block {} has no successors", block_id);
        } else if successors.len() == 1 {
            let next = *successors.iter().next().unwrap();
            println!(
                "[SINGLE_SUCCESSOR] Block {} -> block {}, iterations {}",
                block_id, next, *scope_iterations
            );
            count_instructions_recursive(
                next,
                cfg,
                visited,
                total_instructions,
                scope_instructions,
                scope_iterations,
                instruction_occurrences,
            );
        } else {
            // Two (or more) successors: determine if this is a loop header and
            // which successor is the loop-body continuation vs. the exit.
            let loop_info = cfg.loops.iter().find(|l| l.header == block_id);
            let is_loop = loop_info.is_some();
            let branch_iterations_count: u64 =
                loop_info.map_or(1, |l| l.min_iters as u64);

            let successor_a_id = *successors.iter().nth(0).unwrap();
            let successor_b_id = *successors.iter().nth(1).unwrap();

            // The successor inside the loop body executes `branch_iterations_count`
            // more times; the exit successor runs at the current scope rate.
            let a_in_body = loop_info.map_or(false, |l| l.body.contains(&successor_a_id));
            let b_in_body = loop_info.map_or(false, |l| l.body.contains(&successor_b_id));

            let mut block_a_scope_iterations = if is_loop && a_in_body && !b_in_body {
                *scope_iterations * branch_iterations_count
            } else {
                *scope_iterations
            };
            let mut block_b_scope_iterations = if is_loop && b_in_body && !a_in_body {
                *scope_iterations * branch_iterations_count
            } else {
                *scope_iterations
            };

            if is_loop {
                println!(
                    "[LOOP_BRANCH] Loop header block {}: body-successor={}, exit-successor={}, iters={}",
                    block_id,
                    if a_in_body && !b_in_body { successor_a_id } else { successor_b_id },
                    if a_in_body && !b_in_body { successor_b_id } else { successor_a_id },
                    branch_iterations_count
                );
            } else {
                println!(
                    "[BRANCH] Block {} -> ({}, {}), iterations {}",
                    block_id, successor_a_id, successor_b_id, *scope_iterations
                );
            }

            let mut block_a_scope_instructions = 0;
            let mut block_b_scope_instructions = 0;

            count_instructions_recursive(
                successor_a_id,
                cfg,
                visited,
                total_instructions,
                &mut block_a_scope_instructions,
                &mut block_a_scope_iterations,
                instruction_occurrences,
            );

            count_instructions_recursive(
                successor_b_id,
                cfg,
                visited,
                total_instructions,
                &mut block_b_scope_instructions,
                &mut block_b_scope_iterations,
                instruction_occurrences,
            );

            *total_instructions += *scope_instructions * *scope_iterations;
            println!(
                "[RETURN_SCOPE] Incrementing total instructions by {}, returning from block {}",
                *scope_instructions * *scope_iterations,
                block_id
            );
        }
    }
}

/// Analyze the CFG to produce an instruction count estimate weighted by
/// thread count and loop iteration bounds from `cfg.loops`.
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

    let mut visited = BTreeSet::new();
    let mut total_instructions: u64 = 0;
    let mut scope_instructions: u64 = 0;
    let mut scope_iterations: u64 = 1;
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
        cfg,
        &mut visited,
        &mut total_instructions,
        &mut scope_instructions,
        &mut scope_iterations,
        &mut instruction_occurrences,
    );

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
    }
}
