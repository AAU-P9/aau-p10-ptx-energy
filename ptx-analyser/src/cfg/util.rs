use ptx_parser::r#type::instruction::Inst;
use ptx_parser::r#type::{GeneralOperand, Instruction, Operand};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::common::{BasicBlock, BlockId};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Insert a directed edge and update both adjacency maps.
pub fn add_edge(
    from: BlockId,
    to: BlockId,
    successors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    predecessors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
) {
    successors.entry(from).or_default().insert(to);
    predecessors.entry(to).or_default().insert(from);
}

/// Determine whether an `Inst` is a control-flow terminator.
pub fn is_terminator_inst(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::BraUni(_)
            | Inst::BraUni1(_)
            | Inst::RetUni(_)
            | Inst::Exit(_)
            | Inst::BrxIdxUni(_)
            | Inst::BrxIdxUni1(_)
    )
}

/// Determine whether an `Inst` is any kind of PTX call instruction.
pub fn is_call_inst(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::CallUni(_)
            | Inst::CallUni1(_)
            | Inst::CallUni2(_)
            | Inst::CallUni3(_)
            | Inst::CallUni4(_)
            | Inst::CallUni5(_)
            | Inst::CallUni6(_)
            | Inst::CallUni7(_)
            | Inst::CallUni8(_)
    )
}

/// Returns `true` if the instruction is a return or exit.
pub fn is_return_or_exit(inst: &Inst) -> bool {
    matches!(inst, Inst::RetUni(_) | Inst::Exit(_))
}

/// Try to extract the branch target label name from a `bra` instruction.
pub fn extract_bra_target(inst: &Inst) -> Option<String> {
    let operand = match inst {
        Inst::BraUni(bra) => &bra.tgt,
        Inst::BraUni1(bra) => &bra.tgt,
        _ => return None,
    };
    extract_symbol_name(operand)
}

/// Extract a label / symbol name from a `GeneralOperand`.
pub fn extract_symbol_name(operand: &GeneralOperand) -> Option<String> {
    match operand {
        GeneralOperand::Single { operand, .. } => match operand {
            Operand::Symbol { name, .. } => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Compute outgoing edges for a block whose last instruction is `instruction`.
pub fn add_edges_for_instruction(
    instruction: &Instruction,
    block_idx: BlockId,
    blocks: &[BasicBlock],
    label_to_block: &HashMap<String, BlockId>,
    successors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    predecessors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    exits: &mut Vec<BlockId>,
) {
    let inst = &instruction.inst;
    let is_predicated = instruction.predicate.is_some();

    // ---- Return / Exit ----
    if is_return_or_exit(inst) {
        exits.push(block_idx);
        // A predicated ret/exit can also fall through.
        if is_predicated && block_idx + 1 < blocks.len() {
            add_edge(block_idx, block_idx + 1, successors, predecessors);
        }
        return;
    }

    // ---- Bra (direct branch) ----
    if let Some(target_label) = extract_bra_target(inst) {
        if let Some(&target_block) = label_to_block.get(&target_label) {
            add_edge(block_idx, target_block, successors, predecessors);
        }
        // Conditional (predicated) branch also falls through.
        if is_predicated {
            if block_idx + 1 < blocks.len() {
                add_edge(block_idx, block_idx + 1, successors, predecessors);
            }
        }
        return;
    }

    // ---- Indirect branch (brx.idx) ----
    if matches!(inst, Inst::BrxIdxUni(_) | Inst::BrxIdxUni1(_)) {
        // We cannot statically resolve targets – conservatively, we could
        // connect to all blocks or leave it opaque.  For now mark as exit
        // so the user knows this needs special handling.
        exits.push(block_idx);
        if is_predicated && block_idx + 1 < blocks.len() {
            add_edge(block_idx, block_idx + 1, successors, predecessors);
        }
        return;
    }

    // ---- Non-terminator (normal instruction) – fall through ----
    if block_idx + 1 < blocks.len() {
        add_edge(block_idx, block_idx + 1, successors, predecessors);
    } else {
        // Last block with no explicit terminator – implicit exit.
        exits.push(block_idx);
    }
}
