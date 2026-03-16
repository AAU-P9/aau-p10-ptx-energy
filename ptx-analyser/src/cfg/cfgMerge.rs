use crate::cfg::common::{BasicBlock, BlockId, ControlFlowGraph};
use ptx_parser::r#type::instruction::Inst;
use ptx_parser::r#type::{FunctionStatement, GeneralOperand, Operand, StatementDirective};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone)]
pub struct CallSite {
    pub call_block: BlockId,
    pub callee_entry: BlockId,
    pub callee_return_blocks: Vec<BlockId>,
    pub caller_return_blocks: Vec<BlockId>,
    pub callee_name: String,
    pub inlined_blocks: Vec<BlockId>,
}

#[derive(Debug, Clone)]
pub struct MergedCfg {
    pub cfg: ControlFlowGraph,
    pub call_sites: Vec<CallSite>,
}

fn add_edge(
    successors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    predecessors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    from: BlockId,
    to: BlockId,
) {
    successors.entry(from).or_default().insert(to);
    predecessors.entry(to).or_default().insert(from);
}

fn remove_edge(
    successors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    predecessors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    from: BlockId,
    to: BlockId,
) {
    if let Some(tos) = successors.get_mut(&from) {
        tos.remove(&to);
    }
    if let Some(froms) = predecessors.get_mut(&to) {
        froms.remove(&from);
    }
}

fn ensure_block_slots(cfg: &mut ControlFlowGraph, block_id: BlockId) {
    cfg.successors.entry(block_id).or_default();
    cfg.predecessors.entry(block_id).or_default();
}

fn extract_symbol_name(operand: &GeneralOperand) -> Option<String> {
    match operand {
        GeneralOperand::Single { operand, .. } => match operand {
            Operand::Symbol { name, .. } => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn find_call_target(block: &BasicBlock) -> Option<String> {
    block.statements.iter().rev().find_map(|stmt| {
        match stmt {
            FunctionStatement::Instruction { instruction, .. } => match &instruction.inst {
                Inst::CallUni(call) => extract_symbol_name(&call.func),
                Inst::CallUni1(call) => extract_symbol_name(&call.func),
                Inst::CallUni2(call) => extract_symbol_name(&call.func),
                _ => None,
            },
            FunctionStatement::Directive {
                directive: StatementDirective::CallTargets { directive, .. },
                ..
            } if directive.targets.len() == 1 => directive.targets.first().map(|target| target.val.clone()),
            _ => None,
        }
    })
}

fn last_instruction(block: &BasicBlock) -> Option<&Inst> {
    block.statements.iter().rev().find_map(|stmt| {
        let FunctionStatement::Instruction { instruction, .. } = stmt else {
            return None;
        };
        Some(&instruction.inst)
    })
}

fn is_return_block(block: &BasicBlock) -> bool {
    matches!(last_instruction(block), Some(Inst::RetUni(_)))
}

fn recompute_exits(cfg: &mut ControlFlowGraph) {
    cfg.exits = cfg
        .blocks
        .iter()
        .filter(|block| cfg.successors.get(&block.id).is_none_or(BTreeSet::is_empty))
        .map(|block| block.id)
        .collect();
}

fn inline_call_site(merged: &mut MergedCfg, call_block_id: BlockId, callee_cfg: &ControlFlowGraph) {
    let offset = merged.cfg.blocks.len();

    let remapped_blocks: Vec<BasicBlock> = callee_cfg
        .blocks
        .iter()
        .map(|block| BasicBlock {
            id: block.id + offset,
            label: block
                .label
                .as_ref()
                .map(|label| format!("{}::{}@{}", callee_cfg.function_name, label, offset)),
            statements: block.statements.clone(),
            meta: vec![],
            absorbed_trampoline: false,
        })
        .collect();

    let inlined_blocks: Vec<BlockId> = remapped_blocks.iter().map(|block| block.id).collect();
    let callee_entry = callee_cfg.entry + offset;
    let callee_return_blocks: Vec<BlockId> = callee_cfg
        .blocks
        .iter()
        .filter(|block| is_return_block(block))
        .map(|block| block.id + offset)
        .collect();
    let caller_return_blocks: Vec<BlockId> = merged
        .cfg
        .successors
        .get(&call_block_id)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();

    for block in &remapped_blocks {
        ensure_block_slots(&mut merged.cfg, block.id);
    }

    merged.cfg.blocks.extend(remapped_blocks);

    for (from, tos) in &callee_cfg.successors {
        for to in tos {
            add_edge(
                &mut merged.cfg.successors,
                &mut merged.cfg.predecessors,
                from + offset,
                to + offset,
            );
        }
    }

    for target in &caller_return_blocks {
        remove_edge(
            &mut merged.cfg.successors,
            &mut merged.cfg.predecessors,
            call_block_id,
            *target,
        );
    }

    add_edge(
        &mut merged.cfg.successors,
        &mut merged.cfg.predecessors,
        call_block_id,
        callee_entry,
    );

    for exit_block in &callee_return_blocks {
        for target in &caller_return_blocks {
            add_edge(
                &mut merged.cfg.successors,
                &mut merged.cfg.predecessors,
                *exit_block,
                *target,
            );
        }
    }

    merged.call_sites.push(CallSite {
        call_block: call_block_id,
        callee_entry,
        callee_return_blocks,
        caller_return_blocks,
        callee_name: callee_cfg.function_name.clone(),
        inlined_blocks,
    });
}

// Merge all device-function CFGs into the entry kernel CFG by inlining callees at call sites.
pub fn merge_cfgs(cfgs: &[ControlFlowGraph], kernel_name: &str) -> Option<MergedCfg> {
    let root = cfgs.iter().find(|c| c.function_name == kernel_name)?;
    let callee_by_name: HashMap<&str, &ControlFlowGraph> = cfgs
        .iter()
        .filter(|cfg| cfg.function_name != kernel_name)
        .map(|cfg| (cfg.function_name.as_str(), cfg))
        .collect();

    let mut merged = MergedCfg {
        cfg: root.clone(),
        call_sites: Vec::new(),
    };

    let existing_block_ids: Vec<BlockId> = merged.cfg.blocks.iter().map(|block| block.id).collect();
    for block_id in existing_block_ids {
        ensure_block_slots(&mut merged.cfg, block_id);
    }

    let mut block_idx = 0;
    while block_idx < merged.cfg.blocks.len() {
        if let Some(callee_name) = find_call_target(&merged.cfg.blocks[block_idx]) {
            if let Some(callee_cfg) = callee_by_name.get(callee_name.as_str()) {
                inline_call_site(&mut merged, block_idx, callee_cfg);
            }
        }

        block_idx += 1;
    }

    recompute_exits(&mut merged.cfg);
    Some(merged)
}