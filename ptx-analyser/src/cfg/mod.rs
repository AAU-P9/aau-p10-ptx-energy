#[allow(non_snake_case)]
mod cfgMerge;
mod common;
pub mod cfg;
pub mod ddg;
mod html;
mod util;
pub mod count_instructions;

#[allow(unused_imports)]
pub use cfgMerge::{CallSite, MergedCfg, merge_cfgs};
#[allow(unused_imports)]
pub use common::{BasicBlock, BlockId, CfgEdge, ControlFlowGraph, LoopInfo, Terminator, statement_opcode};
pub use html::{cfg_to_html, merged_cfg_to_html};
pub use count_instructions::analyze_cfg;

use ptx_parser::r#type::{
    EntryFunctionHeaderDirective, FunctionBody, FunctionDim, FunctionStatement, MetaDirective, Module, ModuleDirective,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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
                absorbed_trampoline: false,
                is_inlined: false,
                inlined_from: None,
            }],
            successors: BTreeMap::new(),
            predecessors: BTreeMap::new(),
            entry: 0,
            exits: vec![0],
            meta: vec![],
            maxntid,
            loops: Vec::new(),
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
            absorbed_trampoline: false,
            is_inlined: false,
            inlined_from: None,
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

        let cfg = ControlFlowGraph {
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
        loops: Vec::new(),
    };

    cfg.compact()
}


impl ControlFlowGraph {
    /// Merge chains of blocks where A has one successor B and B has one
    /// predecessor A. Returns a new compacted CFG.
    pub fn compact(&self) -> ControlFlowGraph {
        // Build a union-find / chain map: for each block, find the head of
        // its chain.
        let mut merged_into: Vec<BlockId> = (0..self.blocks.len()).collect();

        for (a, succs) in &self.successors {
            if succs.len() != 1 {
                continue;
            }
            let b = *succs.iter().next().unwrap();
            // B must have exactly one predecessor too
            if self.predecessors.get(&b).map_or(0, |p| p.len()) != 1 {
                continue;
            }
            // Don't merge a block into itself (self-loop)
            if *a == b {
                continue;
            }
            merged_into[b] = merged_into[*a];
        }

        // Flatten chains: each block maps to the representative (head) of its chain
        for i in 0..merged_into.len() {
            let mut root = i;
            while merged_into[root] != root {
                root = merged_into[root];
            }
            merged_into[i] = root;
        }

        // Group blocks by their chain head, preserving order
        let mut chains: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
        for (id, &head) in merged_into.iter().enumerate() {
            chains.entry(head).or_default().push(id);
        }

        // Assign new contiguous block ids
        let old_to_new: BTreeMap<BlockId, BlockId> = chains
            .keys()
            .enumerate()
            .map(|(new_id, &old_head)| (old_head, new_id))
            .collect();

        // Build new blocks
        let mut new_blocks: Vec<BasicBlock> = Vec::with_capacity(chains.len());
        for (&head, members) in &chains {
            let new_id = old_to_new[&head];
            let label = self.blocks[head].label.clone();

            let mut statements = Vec::new();
            let mut meta = Vec::new();
            for &member in members {
                statements.extend(self.blocks[member].statements.iter().cloned());
                meta.extend(self.blocks[member].meta.iter().cloned());
            }

            new_blocks.push(BasicBlock {
                id: new_id,
                label,
                statements,
                meta,
                absorbed_trampoline: false,
                is_inlined: false,
                inlined_from: None,
            });
        }
        new_blocks.sort_by_key(|b| b.id);

        // Rebuild edges, remapping block ids
        let mut new_successors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
        let mut new_predecessors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();

        for b in &new_blocks {
            new_successors.entry(b.id).or_default();
            new_predecessors.entry(b.id).or_default();
        }

        for (old_from, old_tos) in &self.successors {
            let new_from = old_to_new[&merged_into[*old_from]];
            for old_to in old_tos {
                let new_to = old_to_new[&merged_into[*old_to]];
                // Skip intra-chain edges (they no longer exist after merging)
                if new_from == new_to && merged_into[*old_from] == merged_into[*old_to] {
                    continue;
                }
                new_successors.entry(new_from).or_default().insert(new_to);
                new_predecessors.entry(new_to).or_default().insert(new_from);
            }
        }

        let new_entry = old_to_new[&merged_into[self.entry]];
        let new_exits: Vec<BlockId> = self.exits.iter()
            .map(|e| old_to_new[&merged_into[*e]])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let new_meta: Vec<_> = new_blocks.iter()
            .flat_map(|b| b.meta.iter().cloned())
            .collect();

        ControlFlowGraph {
            source_file: self.source_file.clone(),
            function_name: self.function_name.clone(),
            is_entry: self.is_entry,
            blocks: new_blocks,
            successors: new_successors,
            predecessors: new_predecessors,
            entry: new_entry,
            exits: new_exits,
            meta: new_meta,
            maxntid: self.maxntid.clone(),
            loops: Vec::new(),
        }
    }
}