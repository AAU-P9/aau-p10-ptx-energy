use ptx_parser::PtxUnparser;
use ptx_parser::r#type::{FunctionStatement, GeneralOperand, Module, ModuleDirective, Operand, StatementDirective};
use ptx_parser::r#type::meta::MetaTag;
use ptx_parser::r#type::instruction::Inst;
use ptx_parser::r#type::instruction::mov::section_0::Type as MovType;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::common::{BasicBlock, BlockId, ControlFlowGraph, LoopInfo};

pub fn build_cfg(module: &Module, source_file: &str) -> ControlFlowGraph {
    //grab all stmts from entry + device functions
    let mut all_stmts: Vec<FunctionStatement> = Vec::new();
    for directive in &module.directives {
        match directive {
            ModuleDirective::EntryFunction { directive, .. } => {
                if let Some(body) = &directive.body {
                    all_stmts.extend(body.statements.clone());
                }
            }
            ModuleDirective::FuncFunction { directive, .. } => {
                if let Some(body) = &directive.body {
                    all_stmts.extend(body.statements.clone());
                }
            }
            _ => {}
        }
    }

    // PTX wraps multi-stmt call sequences in Block nodes recurse through them
    // so branches/calls inside are visible to the splitter below (this unfucks the inline calls).
    fn flatten(stmts: Vec<FunctionStatement>, out: &mut Vec<FunctionStatement>) {
        for stmt in stmts {
            match stmt {
                FunctionStatement::Block { statements, .. } => flatten(statements, out),
                other => out.push(other),
            }
        }
    }
    let mut flat: Vec<FunctionStatement> = Vec::new();
    flatten(all_stmts, &mut flat);

    // split on every branch/call/ret/exit — that's the only time we start a new block
    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut current: Vec<FunctionStatement> = Vec::new();

    for stmt in flat {
        let is_end = match &stmt {
            FunctionStatement::Instruction { instruction, .. } => {
                is_branch_or_call(&instruction.inst)
            }
            _ => false,
        };

        current.push(stmt);

        if is_end {
            let id = blocks.len();
            blocks.push(BasicBlock { id, label: None, statements: current, meta: Vec::new(), absorbed_trampoline: false });
            current = Vec::new();
        }
    }

    if !current.is_empty() {
        let id = blocks.len();
        blocks.push(BasicBlock { id, label: None, statements: current, meta: Vec::new(), absorbed_trampoline: false });
    }

    //label -> block_id, needed to resolve branch targets
    let mut label_to_block: HashMap<String, BlockId> = HashMap::new();
    for block in &blocks {
        for stmt in &block.statements {
            if let FunctionStatement::Label { label, .. } = stmt {
                label_to_block.insert(label.val.clone(), block.id);
            }
        }
    }

    // seed empty edge sets so every block has an entry even with no edges
    let mut successors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    let mut predecessors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    for b in &blocks {
        successors.insert(b.id, BTreeSet::new());
        predecessors.insert(b.id, BTreeSet::new());
    }

    let mut exits: Vec<BlockId> = Vec::new();

    //wire edges from each block's terminator
    for block in &blocks {
        let id = block.id;
        let next = id + 1;
        let has_next = next < blocks.len();

        // skip labels/directives, find the actual last instruction
        let last_inst = block.statements.iter().rev().find_map(|s| {
            if let FunctionStatement::Instruction { instruction, .. } = s {
                Some(instruction)
            } else {
                None
            }
        });

        let Some(inst) = last_inst else {
            // no instruction — fall through
            if has_next { add_edge(id, next, &mut successors, &mut predecessors); }
            else { exits.push(id); }
            continue;
        };

        let is_predicated = inst.predicate.is_some();

        match &inst.inst {
            // ret/exit terminates the thread
            Inst::RetUni(_) | Inst::Exit(_) => {
                exits.push(id);
                // predicated so there's still a fall-through path
                if is_predicated && has_next {
                    add_edge(id, next, &mut successors, &mut predecessors);
                }
            }

            //Direct branch.
            Inst::BraUni(b) => {
                let dead = is_predicated && is_dead_predicated_branch(&block.statements);
                if !dead {
                    if let Some(label) = extract_symbol(&b.tgt) {
                        if let Some(&target_id) = label_to_block.get(&label) {
                            add_edge(id, target_id, &mut successors, &mut predecessors);
                        } else {
                            println!("[cfg] warn: branch target '{label}' not found");
                        }
                    }
                }
                if is_predicated && has_next {
                    add_edge(id, next, &mut successors, &mut predecessors);
                }
            }
            Inst::BraUni1(b) => {
                let dead = is_predicated && is_dead_predicated_branch(&block.statements);
                if !dead {
                    if let Some(label) = extract_symbol(&b.tgt) {
                        if let Some(&target_id) = label_to_block.get(&label) {
                            add_edge(id, target_id, &mut successors, &mut predecessors);
                        } else {
                            println!("[cfg] warn: branch target '{label}' not found");
                        }
                    }
                }
                if is_predicated && has_next {
                    add_edge(id, next, &mut successors, &mut predecessors);
                }
            }

            //Indirect branch can't resolve statically.
            Inst::BrxIdxUni(_) | Inst::BrxIdxUni1(_) => {
                exits.push(id);
            }

            // call returns to next block; treat it like fall-through
            Inst::CallUni(_) | Inst::CallUni1(_) | Inst::CallUni2(_) | Inst::CallUni3(_)
            | Inst::CallUni4(_) | Inst::CallUni5(_) | Inst::CallUni6(_) | Inst::CallUni7(_)
            | Inst::CallUni8(_) => {
                if has_next {
                    add_edge(id, next, &mut successors, &mut predecessors);
                } else {
                    exits.push(id);
                }
            }

            //shouldn't really appear as a terminator but handle it anyway
            _ => {
                if has_next { add_edge(id, next, &mut successors, &mut predecessors); }
                else { exits.push(id); }
            }
        }
    }

    let mut cfg = ControlFlowGraph {
        source_file: source_file.to_string(),
        function_name: String::new(),
        is_entry: false,
        blocks,
        successors,
        predecessors,
        entry: 0,
        exits,
        meta: Vec::new(),
        maxntid: None,
        loops: Vec::new(),
    };

    compact(&mut cfg);
    println!("[cfg] {} blocks after compaction, {} exits", cfg.blocks.len(), cfg.exits.len());

    cfg.loops = detect_loops(&cfg);
    println!("[cfg] {} natural loops detected", cfg.loops.len());
    for lp in &cfg.loops {
        let unrolled = if lp.is_unrolled { " (unrolled)" } else { "" };
        println!("[cfg]   loop '{}'{}: header=BB{}, tails={:?}, iters={}..{}, body={} blocks",
            lp.label, unrolled, lp.header, lp.tails, lp.min_iters, lp.max_iters, lp.body.len());
    }

    cfg
}

//TODO move to own file
pub fn cfg_to_html(cfg: &ControlFlowGraph) -> String {
    let loop_palette = [
        "#e64553", "#fe640b", "#40a02b", "#04a5e5",
        "#8839ef", "#dd7878", "#179299", "#df8e1d",
    ];
    // build legend directly from cfg.loops — no need to re-scan statements
    let legend_html: String = cfg.loops.iter().enumerate().map(|(i, lp)| {
        let color = loop_palette[i % loop_palette.len()];
        format!(r#"<span style="display:inline-flex;align-items:center;margin-right:14px"><span style="width:12px;height:12px;background:{color};border-radius:2px;margin-right:5px;flex-shrink:0"></span>{label}</span>"#,
            label = html_esc(&lp.label))
    }).collect();

    let dot = cfg_to_dot(cfg);
    // escape for JS template literal — backtick and $ would break the string
    let dot_escaped = dot.replace('\\', "\\\\").replace('`', "\\`").replace('$', "\\$");

    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>CFG</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #1e1e2e; color: #cdd6f4; font-family: monospace; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
header {{ padding: 8px 20px; background: #181825; border-bottom: 1px solid #313244; font-size: .85em; color: #89b4fa; flex-shrink: 0; }}
#graph {{ flex: 1; overflow: auto; padding: 16px; }}
#graph svg {{ max-width: none; display: block; }}
#err {{ color: #f38ba8; padding: 16px; }}
</style>
</head>
<body>
<header>CFG &mdash; {source} &nbsp;&middot;&nbsp; {total} blocks &nbsp;&middot;&nbsp;
  <span style="color:#a6e3a1">&#9632; exit</span> &nbsp;
  <span style="color:#f9e2af">&#9632; entry</span> &nbsp;
  <span style="color:#f38ba8">dashed = back edge</span>
  {legend}
</header>
<div id="graph"><div id="err"></div></div>
<script type="module">
import {{ Graphviz }} from "https://cdn.jsdelivr.net/npm/@hpcc-js/wasm@2/dist/graphviz.js";
const dot = `{dot_escaped}`;
try {{
  const gv = await Graphviz.load();
  const svg = gv.dot(dot, "svg");
  document.getElementById("graph").innerHTML = svg;
}} catch(e) {{
  document.getElementById("err").textContent = "Render error: " + e;
}}
</script>
</body>
</html>"#,
        source = html_esc(&cfg.source_file),
        total = cfg.blocks.len(),
        dot_escaped = dot_escaped,
        legend = if legend_html.is_empty() { String::new() } else {
            format!("&nbsp;&middot;&nbsp; loops: {legend_html}")
        },
    )
}

fn cfg_to_dot(cfg: &ControlFlowGraph) -> String {
    fn html_esc(s: &str) -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
    }

    let back_edges = find_back_edges(cfg);

    //colours for loop regions — wraps around if there are more than 8 loops
    let loop_palette: &[&str] = &[
        "#e64553", "#fe640b", "#40a02b", "#04a5e5",
        "#8839ef", "#dd7878", "#179299", "#df8e1d",
    ];

    //build display strings from pre-computed loop info on the cfg
    let mut block_loop_color: HashMap<BlockId, (&str, String)> = HashMap::new();
    for (i, lp) in cfg.loops.iter().enumerate() {
        let color = loop_palette[i % loop_palette.len()];
        let unrolled = if lp.is_unrolled { " unrolled" } else { "" };
        let display = format!("{} [{}..{}]{}", lp.label, lp.min_iters, lp.max_iters, unrolled);
        for &bid in &lp.body {
            block_loop_color.entry(bid).or_insert((color, display.clone()));
        }
        block_loop_color.insert(lp.header, (color, display));
    }

    let mut out = String::new();
    out.push_str("digraph CFG {\n");
    out.push_str("  graph [bgcolor=\"#1e1e2e\" rankdir=TB fontname=Courier];\n");
    out.push_str("  node  [fontname=Courier fontsize=10 shape=none margin=0];\n");
    out.push_str("  edge  [color=\"#89b4fa\" arrowsize=0.7];\n\n");

    for block in &cfg.blocks {
        let loop_entry = block_loop_color.get(&block.id);
        let (header_bg, header_fg, row_bg) = if let Some((color, _)) = loop_entry {
            (*color, "#1e1e2e", "#1e1e2e")
        } else if cfg.exits.contains(&block.id) {
            ("#40a02b", "#1e1e2e", "#1e2e1e")
        } else if block.id == cfg.entry {
            ("#df8e1d", "#1e1e2e", "#2e2a1e")
        } else {
            ("#45475a", "#cdd6f4", "#1e1e2e")
        };

        // loop label from cfg.loops; no need to scan statements again
        let loop_label: Option<&String> = loop_entry.map(|(_, display)| display);

        // when a trampoline was absorbed, find the index where it starts
        // (the last unconditional bra.uni in the block)
        let tramp_start = if block.absorbed_trampoline {
            block.statements.iter().enumerate().rev().find_map(|(i, s)| {
                if let FunctionStatement::Instruction { instruction, .. } = s {
                    if instruction.predicate.is_none() && matches!(instruction.inst, Inst::BraUni(_) | Inst::BraUni1(_)) {
                        return Some(i);
                    }
                }
                None
            })
        } else {
            None
        };

        let mut rows = String::new();
        for (i, stmt) in block.statements.iter().enumerate() {
            let text: String = stmt.to_tokens_spaced().iter().map(|t| t.as_str()).collect();
            let line = text.trim();
            if line.is_empty() { continue; }

            let in_tramp = tramp_start.map_or(false, |t| i >= t);

            // separator before the trampoline section
            if tramp_start == Some(i) {
                rows.push_str("<TR><TD BGCOLOR=\"#313244\"><FONT COLOR=\"#585b70\" FACE=\"Courier\" POINT-SIZE=\"8\">&#x21B3; absorbed trampoline</FONT></TD></TR>\n");
            }

            let bg = if in_tramp { "#252535" } else { row_bg };
            let color = if in_tramp {
                "#585b70"
            } else {
                match stmt {
                    FunctionStatement::Instruction { .. }                                        => "#cdd6f4",
                    FunctionStatement::Meta { .. }                                               => "#cba6f7",
                    FunctionStatement::Label { .. }                                              => "#585b70",
                    FunctionStatement::Directive { directive: StatementDirective::Loc {..}, .. } => "#585b70",
                    _                                                                            => "#89b4fa",
                }
            };
            let text = wrap_line(&html_esc(line), 60);
            rows.push_str(&format!(
                "<TR><TD ALIGN=\"LEFT\" BGCOLOR=\"{bg}\"><FONT COLOR=\"{color}\" FACE=\"Courier\">{text}</FONT></TD></TR>\n",
            ));
        }

        //header row — loop blocks get a subtitle showing which loop they belong to
        let header_content = if let Some(lbl) = loop_label {
            format!(
                "<B>&#160;BB{id}&#160;</B><BR/><FONT POINT-SIZE=\"8\">&#x27F3; {lbl}</FONT>",
                id = block.id, lbl = html_esc(lbl),
            )
        } else {
            format!("<B>&#160;BB{id}&#160;</B>", id = block.id)
        };

        out.push_str(&format!(
            "  BB{id} [label=<<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\" BGCOLOR=\"{row_bg}\" COLOR=\"{header_bg}\">\
             <TR><TD BGCOLOR=\"{header_bg}\" ALIGN=\"CENTER\"><FONT COLOR=\"{header_fg}\" FACE=\"Courier\">{header_content}</FONT></TD></TR>\
             {rows}</TABLE>>];\n",
            id = block.id,
        ));
    }

    out.push('\n');

    for (from, tos) in &cfg.successors {
        for to in tos {
            if back_edges.contains(&(*from, *to)) {
                out.push_str(&format!(
                    "  BB{from} -> BB{to} [style=dashed color=\"#f38ba8\" constraint=false];\n"
                ));
            } else {
                out.push_str(&format!("  BB{from} -> BB{to};\n"));
            }
        }
    }

    out.push_str("}\n");
    out
}

fn detect_loops(cfg: &ControlFlowGraph) -> Vec<LoopInfo> {
    use std::collections::{HashSet, VecDeque};

    let back_edges = find_back_edges(cfg);
    let back_edge_headers: HashSet<BlockId> = back_edges.iter().map(|(_, h)| *h).collect();

    // find @META Loop annotation blocks and the loop metadata they carry
    let mut annotation_blocks: Vec<(BlockId, String, u32, u32, bool)> = Vec::new();
    for block in &cfg.blocks {
        for stmt in &block.statements {
            if let FunctionStatement::Meta { directive, .. } = stmt {
                if let MetaTag::Loop { label, min_iters, max_iters, is_unrolled } = &directive.tag {
                    annotation_blocks.push((block.id, label.clone(), *min_iters, *max_iters, *is_unrolled));
                    break;
                }
            }
        }
    }

    let mut loops: Vec<LoopInfo> = Vec::new();
    let mut seen_headers: HashSet<BlockId> = HashSet::new();

    for (ann_block, label, min_iters, max_iters, is_unrolled) in annotation_blocks {
        // BFS forward from annotation to nearest back-edge header
        let mut found_header: Option<BlockId> = None;
        let mut visited: HashSet<BlockId> = HashSet::new();
        let mut queue: VecDeque<BlockId> = VecDeque::new();
        queue.push_back(ann_block);
        'bfs: while let Some(cur) = queue.pop_front() {
            if let Some(succs) = cfg.successors.get(&cur) {
                for &s in succs {
                    if back_edge_headers.contains(&s) {
                        found_header = Some(s);
                        break 'bfs;
                    }
                    if visited.insert(s) { queue.push_back(s); }
                }
            }
        }

        let Some(header) = found_header else { continue };
        if seen_headers.contains(&header) { continue; }
        seen_headers.insert(header);

        let tails: Vec<BlockId> = back_edges.iter()
            .filter(|(_, h)| *h == header)
            .map(|(t, _)| *t)
            .collect();

        let lo = tails.iter().copied().min().unwrap_or(header).min(header);
        let hi = tails.iter().copied().max().unwrap_or(header).max(header);
        let body: HashSet<BlockId> = (lo..=hi).collect();

        loops.push(LoopInfo { header, tails, body, label, min_iters, max_iters, is_unrolled });
    }

    loops
}

fn compact(cfg: &mut ControlFlowGraph) {
    use std::collections::HashSet;
    let mut deleted: HashSet<BlockId> = HashSet::new();

    //wtf is this syntax lol
    loop {
        
        // block A ends with unconditional bra.uni, one succ B, B has one pred A
        // -> safe to merge A+B into A
        let candidate = cfg.blocks.iter()
            .filter(|b| !deleted.contains(&b.id))
            .find_map(|block| {
                let last = block.statements.iter().rev().find_map(|s| {
                    if let FunctionStatement::Instruction { instruction, .. } = s { Some(instruction) } else { None }
                })?;
                if last.predicate.is_some() { return None; }
                if !matches!(last.inst, Inst::BraUni(_) | Inst::BraUni1(_)) { return None; }

                let succs = cfg.successors.get(&block.id)?;
                if succs.len() != 1 { return None; }
                let &b_id = succs.iter().next()?;
                if b_id == block.id || deleted.contains(&b_id) { return None; }

                let preds = cfg.predecessors.get(&b_id)?;
                if preds.len() != 1 { return None; }

                Some((block.id, b_id))
            });

        let Some((a_id, b_id)) = candidate else { break };

        //strip the unconditional bra.uni — it's now a no-op after merge
        let a = &mut cfg.blocks[a_id];
        a.statements.retain(|s| !matches!(s,
            FunctionStatement::Instruction { instruction, .. }
            if matches!(instruction.inst, Inst::BraUni(_) | Inst::BraUni1(_))
                && instruction.predicate.is_none()
        ));

        //move B's stmts into A
        let b_stmts = cfg.blocks[b_id].statements.clone();
        cfg.blocks[a_id].statements.extend(b_stmts);

        // A now points wherever B pointed; update predecessor lists of B's succs
        let b_succs = cfg.successors.remove(&b_id).unwrap_or_default();
        cfg.successors.insert(a_id, b_succs.clone());
        cfg.predecessors.remove(&b_id);
        for &c in &b_succs {
            if let Some(preds) = cfg.predecessors.get_mut(&c) {
                preds.remove(&b_id);
                preds.insert(a_id);
            }
        }

        //if B was an exit, A inherits that
        if cfg.exits.contains(&b_id) {
            cfg.exits.retain(|&x| x != b_id);
            if !cfg.exits.contains(&a_id) { cfg.exits.push(a_id); }
        }

        deleted.insert(b_id);
    }

    // Second pass: trampoline absorption — a block whose only real instruction is
    // an unconditional bra.uni and that has exactly one predecessor gets merged
    // into that predecessor. Statements are kept (visible in the parent block)
    // and the parent is flagged with absorbed_trampoline so the visualiser can
    // render the section differently.
    loop {
        let trampoline = cfg.blocks.iter()
            .filter(|b| !deleted.contains(&b.id) && b.id != cfg.entry)
            .find_map(|block| {
                let real_insts: Vec<_> = block.statements.iter().filter_map(|s| {
                    if let FunctionStatement::Instruction { instruction, .. } = s { Some(instruction) } else { None }
                }).collect();
                if real_insts.len() != 1 { return None; }
                let only = real_insts[0];
                if only.predicate.is_some() { return None; }
                if !matches!(only.inst, Inst::BraUni(_) | Inst::BraUni1(_)) { return None; }

                let succs = cfg.successors.get(&block.id)?;
                if succs.len() != 1 { return None; }
                let &target = succs.iter().next()?;
                if target == block.id || deleted.contains(&target) { return None; }

                // only merge if there's exactly one predecessor to absorb into
                let preds = cfg.predecessors.get(&block.id)?;
                if preds.len() != 1 { return None; }
                let &parent = preds.iter().next()?;
                if deleted.contains(&parent) { return None; }

                Some((block.id, target, parent))
            });

        let Some((tramp_id, target_id, parent_id)) = trampoline else { break };

        // append trampoline's statements into the parent so they stay visible
        let tramp_stmts = cfg.blocks[tramp_id].statements.clone();
        cfg.blocks[parent_id].statements.extend(tramp_stmts);
        cfg.blocks[parent_id].absorbed_trampoline = true;

        // rewire: parent now goes to target instead of trampoline
        cfg.successors.get_mut(&parent_id).map(|s| { s.remove(&tramp_id); s.insert(target_id); });
        cfg.predecessors.get_mut(&target_id).map(|p| { p.remove(&tramp_id); p.insert(parent_id); });
        cfg.predecessors.remove(&tramp_id);
        cfg.successors.remove(&tramp_id);

        if cfg.exits.contains(&tramp_id) {
            cfg.exits.retain(|&x| x != tramp_id);
            if !cfg.exits.contains(&parent_id) { cfg.exits.push(parent_id); }
        }

        deleted.insert(tramp_id);
    }

    //compact the block vec — remove deleted slots and re-number from 0
    let old_blocks: Vec<BasicBlock> = std::mem::take(&mut cfg.blocks);
    let id_map: HashMap<BlockId, BlockId> = old_blocks.iter()
        .filter(|b| !deleted.contains(&b.id))
        .enumerate()
        .map(|(new_id, b)| (b.id, new_id))
        .collect();

    cfg.blocks = old_blocks.into_iter()
        .filter(|b| !deleted.contains(&b.id))
        .enumerate()
        .map(|(new_id, mut b)| { b.id = new_id; b })
        .collect();

    let remap = |id: BlockId| *id_map.get(&id).unwrap_or(&id);

    cfg.successors = cfg.successors.iter()
        .filter(|(id, _)| !deleted.contains(id))
        .map(|(id, succs)| (remap(*id), succs.iter().map(|s| remap(*s)).collect()))
        .collect();

    cfg.predecessors = cfg.predecessors.iter()
        .filter(|(id, _)| !deleted.contains(id))
        .map(|(id, preds)| (remap(*id), preds.iter().map(|p| remap(*p)).collect()))
        .collect();

    cfg.entry = remap(cfg.entry);
    cfg.exits = cfg.exits.iter().map(|&e| remap(e)).collect();
}

fn find_back_edges(cfg: &ControlFlowGraph) -> std::collections::HashSet<(BlockId, BlockId)> {
    use std::collections::HashSet;

    let mut back_edges: HashSet<(BlockId, BlockId)> = HashSet::new();
    let mut visited: HashSet<BlockId> = HashSet::new();

    //one DFS per connected component — sharing in_stack across components gives false positives
    for start in cfg.blocks.iter().map(|b| b.id) {
        if visited.contains(&start) { continue; }

        let mut in_stack: HashSet<BlockId> = HashSet::new();
        let mut stack: Vec<(BlockId, bool)> = vec![(start, false)];

        while let Some((node, returning)) = stack.pop() {
            if returning {
                in_stack.remove(&node);
                continue;
            }
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node);
            in_stack.insert(node);
            stack.push((node, true));

            if let Some(succs) = cfg.successors.get(&node) {
                for &next in succs {
                    if in_stack.contains(&next) {
                        back_edges.insert((node, next));
                    } else if !visited.contains(&next) {
                        stack.push((next, false));
                    }
                }
            }
        }
    }

    back_edges
}

fn html_esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn wrap_line(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut remaining = s;
    while remaining.len() > max {
        out.push_str(&remaining[..max]);
        out.push_str("<BR/>");
        remaining = &remaining[max..];
    }
    out.push_str(remaining);
    out
}

fn add_edge(
    from: BlockId,
    to: BlockId,
    successors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
    predecessors: &mut BTreeMap<BlockId, BTreeSet<BlockId>>,
) {
    successors.entry(from).or_default().insert(to);
    predecessors.entry(to).or_default().insert(from);
}

fn extract_symbol(operand: &GeneralOperand) -> Option<String> {
    match operand {
        GeneralOperand::Single { operand, .. } => match operand {
            Operand::Symbol { name, .. } => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a predicated branch is statically dead — i.e. the predicate register
/// was set to a constant that makes the branch never taken.
///
/// Matches the pattern emitted by CUDA for ASSUME_DIM/ASSUME_RANGE constraints:
///   mov.pred %pX, 0;   @%pX  bra target;  — branch never taken
///   mov.pred %pX, 1;   @!%pX bra target;  — same, negated form
fn is_dead_predicated_branch(stmts: &[FunctionStatement]) -> bool {
    // collect instructions only, in order
    let insts: Vec<_> = stmts.iter().filter_map(|s| {
        if let FunctionStatement::Instruction { instruction, .. } = s { Some(instruction) } else { None }
    }).collect();

    let (Some(branch), Some(prev)) = (insts.last(), insts.iter().rev().nth(1)) else {
        return false;
    };

    // the branch must be predicated
    let Some(pred) = &branch.predicate else { return false };

    // get branch predicate register name
    let pred_reg = match &pred.operand {
        Operand::Register { operand, .. } => &operand.name,
        _ => return false,
    };

    // previous instruction must be mov.pred %pX, <imm>
    let Inst::MovType(mov) = &prev.inst else { return false };
    if mov.type_ != MovType::Pred { return false; }

    // get destination register of the mov
    let dst_reg = match &mov.d {
        GeneralOperand::Single { operand: Operand::Register { operand, .. }, .. } => &operand.name,
        _ => return false,
    };
    if dst_reg != pred_reg { return false; }

    // get immediate value from mov source
    let imm_val = match &mov.a {
        GeneralOperand::Single { operand: Operand::Immediate { operand, .. }, .. } => operand.value.trim_end_matches('U'),
        _ => return false,
    };

    // dead if: mov 0 + not-negated  OR  mov 1 + negated
    let mov_zero = imm_val == "0";
    let mov_one  = imm_val == "1";
    (mov_zero && !pred.negated) || (mov_one && pred.negated)
}

fn is_branch_or_call(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::BraUni(_)
            | Inst::BraUni1(_)
            | Inst::BrxIdxUni(_)
            | Inst::BrxIdxUni1(_)
            | Inst::RetUni(_)
            | Inst::Exit(_)
            | Inst::CallUni(_)
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
