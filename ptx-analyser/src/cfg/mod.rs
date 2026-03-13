#[allow(non_snake_case)]
mod cfgMerge;
mod common;
mod html;
mod util;

#[allow(unused_imports)]
pub use cfgMerge::{CallSite, MergedCfg, merge_cfgs};
#[allow(unused_imports)]
pub use common::{BasicBlock, BlockId, CfgEdge, ControlFlowGraph, Terminator};
pub use html::{cfg_to_html, merged_cfg_to_html};

use ptx_parser::r#type::{
    EntryFunctionHeaderDirective, FunctionBody, FunctionDim, FunctionStatement, MetaDirective,
    Module, ModuleDirective, Operand,
    instruction::{Inst, ld::LdWeakSsCopLevelCacheHintLevelPrefetchSizeVecType},
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use crate::{Parameter, cfg::common::format_inst_short};

use util::{add_edge, add_edges_for_instruction, is_call_inst, is_terminator_inst};

// ---------------------------------------------------------------------------
// Builder – construct a CFG from the flat list of FunctionStatements
// ---------------------------------------------------------------------------

/// Build CFGs for every function / entry in the module.
pub fn build_cfgs(module: &Module, source_file: &str) -> Vec<ControlFlowGraph> {
    let mut cfgs = Vec::new();

    for dir in &module.directives {
        match dir {
            ModuleDirective::EntryFunction { directive, .. } => {
                if let Some(body) = &directive.body {
                    let name = directive.name.val.clone();
                    let maxntid = extract_maxntid(&directive.directives);
                    cfgs.push(build_cfg_with_header(
                        &name,
                        body,
                        source_file,
                        maxntid,
                        true,
                    ));
                }
            }
            ModuleDirective::FuncFunction { directive, .. } => {
                if let Some(body) = &directive.body {
                    let name = directive.name.val.clone();
                    cfgs.push(build_cfg(&name, body, source_file));
                }
            }
            _ => {}
        }
    }

    cfgs
}

/// Extract the `.maxntid` header directive (if any) from an entry's header
/// directive list.
fn extract_maxntid(directives: &[EntryFunctionHeaderDirective]) -> Option<FunctionDim> {
    directives.iter().find_map(|d| match d {
        EntryFunctionHeaderDirective::MaxNTid { dim, .. } => Some(dim.clone()),
        _ => None,
    })
}

/// Build a single CFG from a function body (no header information).
///
/// This is used for plain `.func` device functions that do not carry
/// launch-bounds style directives such as `.maxntid`.
pub fn build_cfg(function_name: &str, body: &FunctionBody, source_file: &str) -> ControlFlowGraph {
    build_cfg_with_header(function_name, body, source_file, None, false)
}

/// Internal helper that also takes entry-function header information such as
/// `.maxntid` (if available).
fn build_cfg_with_header(
    function_name: &str,
    body: &FunctionBody,
    source_file: &str,
    maxntid: Option<FunctionDim>,
    is_entry: bool,
) -> ControlFlowGraph {
    let stmts = &body.statements;

    if stmts.is_empty() {
        return ControlFlowGraph {
            source_file: source_file.to_string(),
            function_name: function_name.to_string(),
            is_entry,
            blocks: vec![BasicBlock {
                id: 0,
                label: None,
                statements: vec![],
                meta: vec![],
            }],
            successors: BTreeMap::new(),
            predecessors: BTreeMap::new(),
            entry: 0,
            exits: vec![0],
            meta: vec![],
            maxntid,
        };
    }

    // ------------------------------------------------------------------
    // Pass 1 – identify leader indices.
    //
    // A leader is:
    //   * index 0 (the first statement)
    //   * any Label statement
    //   * any statement immediately following a branch / ret / exit
    // ------------------------------------------------------------------
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    leaders.insert(0);

    // We also collect the set of target label names so that we can later
    // resolve them to block ids.  And we note which indices are "after a
    // terminator" so that the next statement is a leader.
    for (i, stmt) in stmts.iter().enumerate() {
        match stmt {
            FunctionStatement::Label { .. } => {
                leaders.insert(i);
            }
            FunctionStatement::Instruction { instruction, .. } => {
                if is_call_inst(&instruction.inst) {
                    if i > 0 {
                        leaders.insert(i);
                    }
                    if i + 1 < stmts.len() {
                        leaders.insert(i + 1);
                    }
                }

                if is_terminator_inst(&instruction.inst) {
                    // The statement *after* this one starts a new block.
                    if i + 1 < stmts.len() {
                        leaders.insert(i + 1);
                    }
                }
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------------
    // Pass 2 – carve the statements into basic blocks.
    // ------------------------------------------------------------------
    let leader_vec: Vec<usize> = leaders.iter().copied().collect();
    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(leader_vec.len());

    for (block_idx, &start) in leader_vec.iter().enumerate() {
        let end = leader_vec
            .get(block_idx + 1)
            .copied()
            .unwrap_or(stmts.len());

        let block_stmts: Vec<FunctionStatement> = stmts[start..end]
            .iter()
            .filter(|s| !matches!(s, FunctionStatement::Meta { .. }))
            .cloned()
            .collect();

        let block_meta: Vec<MetaDirective> = stmts[start..end]
            .iter()
            .filter_map(|s| match s {
                FunctionStatement::Meta { directive, .. } => Some(directive.clone()),
                _ => None,
            })
            .collect();

        let label = block_stmts.first().and_then(|s| match s {
            FunctionStatement::Label { label, .. } => Some(label.val.clone()),
            _ => None,
        });

        blocks.push(BasicBlock {
            id: block_idx,
            label,
            statements: block_stmts,
            meta: block_meta,
        });
    }

    // Derive cfg-level meta from all blocks.
    let meta: Vec<MetaDirective> = blocks.iter().flat_map(|b| b.meta.iter().cloned()).collect();

    // Map from label name → block id for resolving branch targets.
    let label_to_block: HashMap<String, BlockId> = blocks
        .iter()
        .filter_map(|b| b.label.as_ref().map(|l| (l.clone(), b.id)))
        .collect();

    // ------------------------------------------------------------------
    // Pass 3 – compute edges.
    // ------------------------------------------------------------------
    let mut successors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    let mut predecessors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    let mut exits: Vec<BlockId> = Vec::new();

    // Ensure every block has an entry in the maps even if it has no edges.
    for b in &blocks {
        successors.entry(b.id).or_default();
        predecessors.entry(b.id).or_default();
    }

    for (block_idx, block) in blocks.iter().enumerate() {
        // Find the last instruction in the block (skip trailing labels /
        // directives that don't affect control flow).
        let last_inst = block.statements.iter().rev().find_map(|s| match s {
            FunctionStatement::Instruction { instruction, .. } => Some(instruction),
            _ => None,
        });

        match last_inst {
            Some(instruction) => {
                add_edges_for_instruction(
                    instruction,
                    block_idx,
                    &blocks,
                    &label_to_block,
                    &mut successors,
                    &mut predecessors,
                    &mut exits,
                );
            }
            None => {
                // Block has no instructions (e.g., only labels / directives).
                // Falls through to the next block.
                if block_idx + 1 < blocks.len() {
                    add_edge(block_idx, block_idx + 1, &mut successors, &mut predecessors);
                } else {
                    exits.push(block_idx);
                }
            }
        }
    }

    // Deduplicate exits.
    exits.sort_unstable();
    exits.dedup();

    ControlFlowGraph {
        source_file: source_file.to_string(),
        function_name: function_name.to_string(),
        is_entry,
        blocks,
        successors,
        predecessors,
        entry: 0,
        exits,
        meta,
        maxntid,
    }
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
    previous_block_predecessor_count: usize,
    instruction_occurrences: &mut HashMap<String, u64>,
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
            *scope_instructions += 1;
            instruction_occurrences.entry(format_inst_short(&instruction.inst))
                .and_modify(|count| *count += *scope_iterations)
                .or_insert(*scope_iterations);
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
                block_predecessor_count,
                instruction_occurrences,
            );
        } else if successors.len() > 1 {
            let branch_info = block.get_cmp_info(registers).unwrap();
            let is_loop = previous_block_predecessor_count > 1;

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
                    branch_info.iterations_count
                );
            } else {
                println!(
                    "[BRANCH] Branch found at block {}: true branch is {}, iteration count is {}",
                    block_id,
                    if is_block_a_true_branch { "A" } else { "B" },
                    branch_info.iterations_count
                );
            }

            let mut block_a_scope_instructions = 0;
            let mut block_b_scope_instructions = 0;

            if is_block_a_true_branch {
                let mut block_a_scope_iterations = *scope_iterations;
                let mut block_b_scope_iterations = if is_loop {
                    *scope_iterations * branch_info.iterations_count
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
                    block_predecessor_count,
                    instruction_occurrences,
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
                    block_predecessor_count,
                    instruction_occurrences,
                );
            } else {
                let mut block_a_scope_iterations = if is_loop {
                    *scope_iterations * branch_info.iterations_count
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
                    block_predecessor_count,
                    instruction_occurrences,
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
                    block_predecessor_count,
                    instruction_occurrences,
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
pub fn analyze_cfgs(
    cfgs: &Vec<ControlFlowGraph>,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    params: &Vec<Parameter>,
) {
    let root_name = cfgs
        .iter()
        .find(|cfg| cfg.is_entry)
        .map(|cfg| cfg.function_name.as_str())
        .unwrap_or_else(|| cfgs[0].function_name.as_str());
    let cfg = merge_cfgs(cfgs, root_name)
        .map(|merged| merged.cfg)
        .unwrap_or_else(|| cfgs[0].clone());

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
        0,
        &mut instruction_occurrences,
    );
// 
    println!("Total instructions for the CFG: {}", total_instructions);
    println!("Instruction occurrences:");
    for (inst, count) in &instruction_occurrences {
        println!("  - {}: {}", inst, count);
    }
}

// ---------------------------------------------------------------------------
// Unit-style sanity helpers (not exhaustive – useful during development)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ptx_parser::parse_ptx;

    /// Minimal kernel that has a single basic block.
    #[test]
    fn single_block() {
        let src = r#"
            .version 8.0
            .target sm_52
            .address_size 64

            .entry kernel() {
                ret;
            }
        "#;
        let module = parse_ptx(src).expect("parse");
        let cfgs = build_cfgs(&module, "test.ptx");
        assert_eq!(cfgs.len(), 1);
        let cfg = &cfgs[0];
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.exits.len(), 1);
        assert_eq!(cfg.source_file, "test.ptx");
        assert!(cfg.is_entry);
    }

    /// Kernel with an unconditional branch creating two blocks.
    #[test]
    fn unconditional_branch() {
        let src = r#"
            .version 8.0
            .target sm_52
            .address_size 64

            .entry kernel() {
                bra.uni target;
            target:
                ret;
            }
        "#;
        let module = parse_ptx(src).expect("parse");
        let cfgs = build_cfgs(&module, "test.ptx");
        assert_eq!(cfgs.len(), 1);
        let cfg = &cfgs[0];
        // Should have 2 blocks: the bra and the target label block.
        assert!(cfg.blocks.len() >= 2);
        // The entry block should have the target block as successor.
        let entry_succs = cfg.successors.get(&cfg.entry).unwrap();
        assert!(!entry_succs.is_empty());
    }

    #[test]
    fn dot_output_is_nonempty() {
        let src = r#"
            .version 8.0
            .target sm_52
            .address_size 64

            .entry kernel() {
                ret;
            }
        "#;
        let module = parse_ptx(src).expect("parse");
        let cfgs = build_cfgs(&module, "test.ptx");
        let dot = cfgs[0].to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("BB0"));
    }

    #[test]
    fn html_output_includes_source_file() {
        let src = r#"
            .version 8.0
            .target sm_52
            .address_size 64

            .entry kernel() {
                ret;
            }
        "#;
        let module = parse_ptx(src).expect("parse");
        let cfgs = build_cfgs(&module, "my_kernel.ptx");
        let html = cfg_to_html(&cfgs[0]);
        assert!(html.contains("my_kernel.ptx"));
        assert!(html.contains("kernel"));
    }

    #[test]
    fn merge_inlines_single_calltargets_helper() {
        let src = r#"
            .version 8.5
            .target sm_80
            .address_size 64

            .visible .func helper(.param .b32 helper_param)
            {
                .reg .b32 %r0;
                mov.u32 %r0, %r0;
                ret;
            }

            .visible .entry kernel()
            {
            entry_call:
                .calltargets helper;
            entry_proto:
                ret;
            }
        "#;

        let module = parse_ptx(src).expect("parse");
        let cfgs = build_cfgs(&module, "merge_test.ptx");
        let merged = merge_cfgs(&cfgs, "kernel").expect("merged cfg");

        assert_eq!(merged.call_sites.len(), 1);

        let call_site = &merged.call_sites[0];
        let call_successors = merged.cfg.successors.get(&call_site.call_block).unwrap();
        let helper_block = call_site.callee_entry;
        let return_block = call_site.caller_return_blocks[0];

        assert!(call_successors.contains(&helper_block));
        assert!(
            merged
                .cfg
                .successors
                .get(&helper_block)
                .unwrap()
                .contains(&return_block)
        );
    }
}
