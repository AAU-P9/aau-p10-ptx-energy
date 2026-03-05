mod common;
mod html;
mod util;

#[allow(unused_imports)]
pub use common::{BasicBlock, BlockId, CfgEdge, ControlFlowGraph, Terminator};
pub use html::cfg_to_html;

use ptx_parser::r#type::{FunctionBody, FunctionStatement, MetaDirective, Module, ModuleDirective, Operand, instruction::Inst};
use std::{collections::{BTreeMap, BTreeSet, HashMap}, ffi::c_void, thread::scope};

use util::{add_edge, add_edges_for_instruction, is_terminator_inst};

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
                    cfgs.push(build_cfg(&name, body, source_file));
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

/// Build a single CFG from a function body.
pub fn build_cfg(function_name: &str, body: &FunctionBody, source_file: &str) -> ControlFlowGraph {
    let stmts = &body.statements;

    if stmts.is_empty() {
        return ControlFlowGraph {
            source_file: source_file.to_string(),
            function_name: function_name.to_string(),
            blocks: vec![BasicBlock {
                id: 0,
                label: None,
                statements: vec![],
            }],
            successors: BTreeMap::new(),
            predecessors: BTreeMap::new(),
            entry: 0,
            exits: vec![0],
            meta: vec![],
        };
    }

    // ------------------------------------------------------------------
    // Collect @META annotations from the function body.
    // ------------------------------------------------------------------
    let meta: Vec<MetaDirective> = stmts
        .iter()
        .filter_map(|s| match s {
            FunctionStatement::Meta { directive, .. } => Some(directive.clone()),
            _ => None,
        })
        .collect();

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

        let label = block_stmts.first().and_then(|s| match s {
            FunctionStatement::Label { label, .. } => Some(label.val.clone()),
            _ => None,
        });

        blocks.push(BasicBlock {
            id: block_idx,
            label,
            statements: block_stmts,
        });
    }

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
        blocks,
        successors,
        predecessors,
        entry: 0,
        exits,
        meta,
    }
}

#[derive(Copy, Clone)]
pub struct Bound {
    pub min: i128,
    pub max: i128,
}

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

struct BranchInfo {
    target_label: String,
    iteration_count: i64,
}

pub trait IsBranch {
    fn is_branch(&self) -> Option<BranchInfo>;
}   

impl IsBranch for BasicBlock {
    fn is_branch(&self) -> Option<BranchInfo> {
        let mut is_branch = false;
        let mut target_label: String = String::new();
        let mut iteration_count: i64 = 1;

        for stmt in &self.statements {
            if let FunctionStatement::Instruction { instruction, .. } = stmt {
                    if let ptx_parser::r#type::instruction::Inst::SetpCmpopFtzType(inst) = &instruction.inst  {
                        if let ptx_parser::r#type::GeneralOperand::Single { operand, .. } = &inst.b {
                            if let Operand::Immediate { operand, .. } = operand {
                                iteration_count = operand.value.parse().unwrap();
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
            return Some(BranchInfo {
                target_label,
                iteration_count,
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
    total_instructions: &mut i64,
    scope_instructions: &mut i64,
    scope_iterations: &mut i64,
) {
    // Avoid infinite loops by tracking visited blocks
    if visited.contains(&block_id) || cfg.successors.get(&block_id).is_none() {
        *total_instructions += *scope_instructions * *scope_iterations;
        println!("[RETURN_END] Incrementing total instructions by {}, returning from block {}", *scope_instructions * *scope_iterations, block_id);
        return;
    }

    visited.insert(block_id);

    // Count instructions in current block
    let block = &cfg.blocks[block_id];
    *scope_instructions += block.statements.len() as i64;

    // Debug: Print if the block has no instructions, this should be impossible.
    if block.statements.is_empty() {
        println!("[ERROR] Block {} has no statements", block_id);
    }

    println!("[VISIT] Visiting block {}, current scope instructions: {}, current scope iterations: {}", block_id, *scope_instructions, *scope_iterations);

    
    // Recursively count instructions in successor blocks
    if let Some(successors) = cfg.successors.get(&block_id) {
        if successors.len() == 1 {
            count_instructions_recursive(*successors.iter().next().unwrap(), cfg, visited, total_instructions, scope_instructions, scope_iterations);
        } else if successors.len() > 1 {
            let branch_info = block.is_branch().unwrap();

            let successor_a_id = *successors.iter().nth(0).unwrap();
            let successor_b_id = *successors.iter().nth(1).unwrap();

            let successor_a_block = &cfg.blocks[successor_a_id];
            let is_block_a_true_branch = successor_a_block.label.is_some() && successor_a_block.label.as_ref().unwrap() == &branch_info.target_label;
            
            println!("[BRANCH] Branch found at block {}: true branch is {}, iteration count is {}", block_id, if is_block_a_true_branch { "A" } else { "B" }, branch_info.iteration_count);

            let mut block_a_scope_instructions = 0;
            let mut block_b_scope_instructions = 0;
            
            if (is_block_a_true_branch) {
                let mut block_a_scope_iterations = *scope_iterations;
                let mut block_b_scope_iterations = *scope_iterations * branch_info.iteration_count;
                
                
                println!("[A_TRUE_BRANCH] Visiting false branch (block {}) with scope iterations {} from block {}", successor_b_id, block_b_scope_iterations, block_id);
                count_instructions_recursive(successor_b_id, cfg, visited, total_instructions, &mut block_b_scope_instructions, &mut block_b_scope_iterations);

                println!("[A_TRUE_BRANCH] Visiting true branch (block {}) with scope iterations {} from block {}", successor_a_id, block_a_scope_iterations, block_id);
                count_instructions_recursive(successor_a_id, cfg, visited, total_instructions, &mut block_a_scope_instructions, &mut block_a_scope_iterations);
            } else {
                let mut block_a_scope_iterations = *scope_iterations * branch_info.iteration_count;
                let mut block_b_scope_iterations = *scope_iterations;
                
                println!("[A_FALSE_BRANCH] Visiting false branch (block {}) with scope iterations {} from block {}", successor_a_id, block_a_scope_iterations, block_id);
                count_instructions_recursive(successor_a_id, cfg, visited, total_instructions, &mut block_a_scope_instructions, &mut block_a_scope_iterations);

                println!("[A_FALSE_BRANCH] Visiting true branch (block {}) with scope iterations {} from block {}", successor_b_id, block_b_scope_iterations, block_id);
                count_instructions_recursive(successor_b_id, cfg, visited, total_instructions, &mut block_b_scope_instructions, &mut block_b_scope_iterations);

            }

            *total_instructions += *scope_instructions * *scope_iterations;
            println!("[RETURN_SCOPE] Incrementing total instructions by {}, returning from block {}", *scope_instructions * *scope_iterations, block_id);
        }
    }
}

// Analyze the cfg to get a power consumption estimate
pub fn analyze_cfgs(cfgs: &Vec<ControlFlowGraph>) {
    let mut symbol_table: HashMap<String, Bound> = HashMap::new();

    let cfg = &cfgs[0];

    // Count total instructions by recursively walking the CFG tree
    let mut visited = BTreeSet::new();
    let mut total_instructions: i64 = 0;
    let mut scope_instructions: i64 = 0;
    let mut scope_iterations = 1;

    count_instructions_recursive(cfg.entry, cfg, &mut visited, &mut total_instructions, &mut scope_instructions, &mut scope_iterations);

    println!("Total instructions for the CFG: {}", total_instructions);
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
}
