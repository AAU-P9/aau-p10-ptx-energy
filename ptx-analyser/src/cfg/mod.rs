mod common;
mod html;
mod util;

#[allow(unused_imports)]
pub use common::{BasicBlock, BlockId, CfgEdge, ControlFlowGraph, Terminator};
pub use html::cfg_to_html;

use ptx_parser::r#type::{FunctionBody, FunctionStatement, Module, ModuleDirective};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

        let block_stmts: Vec<FunctionStatement> = stmts[start..end].to_vec();

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
