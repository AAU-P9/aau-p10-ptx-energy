use std::collections::HashMap;

use ptx_parser::PtxToken;
use ptx_parser::PtxUnparser;
use ptx_parser::r#type::FunctionStatement;

use super::common::{BlockId, ControlFlowGraph, dot_html_escape, format_stmt_ptx};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identifies a single instruction: (block_id, index among instructions in that block).
pub type InstrId = (BlockId, usize);

/// A RAW (read-after-write) data dependency.
#[derive(Debug, Clone)]
pub struct DdgEdge {
    /// The instruction that writes `register`.
    pub from: InstrId,
    /// The instruction that subsequently reads `register`.
    pub to: InstrId,
    pub register: String,
    /// True when `from` and `to` are in different basic blocks.
    pub cross_block: bool,
}

/// Intra-block data dependency graph for one CFG.
///
/// Nodes are implicit (any `InstrId` appearing in `defs`/`uses`).
/// Edges are intra-block RAW dependencies only — cross-block reaching-defs are
/// not tracked here.
pub struct DataDependencyGraph {
    pub cfg_name: String,
    pub edges: Vec<DdgEdge>,
    /// PTX text of each instruction node (for rendering).
    pub nodes: HashMap<InstrId, String>,
    /// Registers defined (written) by each instruction.
    pub defs: HashMap<InstrId, Vec<String>>,
    /// Registers used (read) by each instruction.
    pub uses: HashMap<InstrId, Vec<String>>,
}

// ---------------------------------------------------------------------------
// Def/use extraction
// ---------------------------------------------------------------------------

/// Extract the registers defined and used by one instruction statement.
///
/// Uses the token stream to avoid matching every `Inst` variant.
///
/// PTX convention applied here:
/// - `@{!}%reg` predicate → **use**
/// - Slot 0 (before first top-level comma):
///   - Starts with `[` → store: **no def**, all registers are **uses**
///   - Otherwise: `Register` tokens at bracket/brace depth 0 → **defs**
///     (covers single dest and dual-dest `p|q` from `setp`)
/// - Slots 1+ → all `Register` tokens → **uses**
/// - `Register` tokens inside `[…]` are always **uses** (address registers)
pub fn extract_defs_uses(stmt: &FunctionStatement) -> (Vec<String>, Vec<String>) {
    if !matches!(stmt, FunctionStatement::Instruction { .. }) {
        return (vec![], vec![]);
    }

    let tokens = stmt.to_tokens_spaced();
    let n = tokens.len();
    let mut i = 0;
    let mut defs: Vec<String> = Vec::new();
    let mut uses: Vec<String> = Vec::new();

    macro_rules! skip_ws {
        () => {
            while i < n && matches!(tokens[i], PtxToken::Space | PtxToken::Newline) {
                i += 1;
            }
        };
    }

    skip_ws!();

    // 1. Optional predicate guard: @[!]%reg → use
    if i < n && matches!(tokens[i], PtxToken::At) {
        i += 1;
        if i < n && matches!(tokens[i], PtxToken::Exclaim) {
            i += 1;
        }
        skip_ws!();
        if i < n {
            if let PtxToken::Register(r) = &tokens[i] {
                uses.push(r.clone());
                i += 1;
            }
        }
        skip_ws!();
    }

    // 2. Skip opcode (identifiers, dots, colons, etc.) until first operand token
    while i < n {
        match &tokens[i] {
            PtxToken::Register(_)
            | PtxToken::LBracket
            | PtxToken::LBrace
            | PtxToken::Semicolon => break,
            _ => i += 1,
        }
    }

    // 3. Is slot 0 a store address (starts with `[`)?
    skip_ws!();
    let slot0_is_address = i < n && matches!(tokens[i], PtxToken::LBracket);

    // 4. Walk operands tracking depth and slot number
    let mut slot: usize = 0;
    let mut bracket_depth: i32 = 0;
    let mut brace_depth: i32 = 0;

    while i < n {
        match &tokens[i] {
            PtxToken::Semicolon => break,
            PtxToken::LBracket => {
                bracket_depth += 1;
                i += 1;
            }
            PtxToken::RBracket => {
                bracket_depth -= 1;
                i += 1;
            }
            PtxToken::LBrace => {
                brace_depth += 1;
                i += 1;
            }
            PtxToken::RBrace => {
                brace_depth -= 1;
                i += 1;
            }
            PtxToken::Comma if bracket_depth == 0 && brace_depth == 0 => {
                slot += 1;
                i += 1;
            }
            PtxToken::Register(r) => {
                let r = r.clone();
                if slot == 0 && !slot0_is_address && bracket_depth == 0 {
                    defs.push(r);
                } else {
                    uses.push(r);
                }
                i += 1;
            }
            _ => i += 1,
        }
    }

    (defs, uses)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a global DDG for `cfg`.
///
/// Pass 1: collect every register definition across all blocks.
/// Pass 2: for every use, look up the defining instruction globally and emit
///         a `DdgEdge`.  Cross-block edges are flagged so they can be styled
///         differently in the visualisation.
///
/// PTX virtual registers are SSA (each written exactly once), so a global
/// def-map is exact — no reaching-defs analysis required.
pub fn build_ddg(cfg: &ControlFlowGraph) -> DataDependencyGraph {
    let mut all_defs: HashMap<InstrId, Vec<String>> = HashMap::new();
    let mut all_uses: HashMap<InstrId, Vec<String>> = HashMap::new();
    let mut nodes: HashMap<InstrId, String> = HashMap::new();

    // Pass 1: map every register name → the single instruction that defines it.
    // Skip instructions with no register operands at all (branches, barriers, etc.)
    // — they contribute nothing to the DDG and only clutter the graph.
    let mut global_def: HashMap<String, InstrId> = HashMap::new();
    for block in &cfg.blocks {
        let mut instr_idx = 0usize;
        for stmt in &block.statements {
            if !matches!(stmt, FunctionStatement::Instruction { .. }) {
                continue;
            }
            let id: InstrId = (block.id, instr_idx);
            let (defs, uses) = extract_defs_uses(stmt);
            instr_idx += 1;
            if defs.is_empty() && uses.is_empty() {
                continue;
            }
            for d in &defs {
                global_def.insert(d.clone(), id);
            }
            nodes.insert(id, format_stmt_ptx(stmt));
            all_defs.insert(id, defs);
            all_uses.insert(id, uses);
        }
    }

    // Pass 2: for every use, emit an edge from the defining instruction.
    let mut edges: Vec<DdgEdge> = Vec::new();
    for block in &cfg.blocks {
        let mut instr_idx = 0usize;
        for stmt in &block.statements {
            if !matches!(stmt, FunctionStatement::Instruction { .. }) {
                continue;
            }
            let to: InstrId = (block.id, instr_idx);
            instr_idx += 1;
            // Nodes absent from all_uses were filtered out in pass 1
            if let Some(uses) = all_uses.get(&to) {
                for u in uses {
                    if let Some(&from) = global_def.get(u.as_str()) {
                        if from != to {
                            edges.push(DdgEdge {
                                from,
                                to,
                                register: u.clone(),
                                cross_block: from.0 != to.0,
                            });
                        }
                    }
                }
            }
        }
    }

    DataDependencyGraph {
        cfg_name: cfg.function_name.clone(),
        edges,
        nodes,
        defs: all_defs,
        uses: all_uses,
    }
}

// ---------------------------------------------------------------------------
// DOT export
// ---------------------------------------------------------------------------

impl DataDependencyGraph {
    /// Render the DDG as a Graphviz DOT string.
    ///
    /// Instructions are grouped into subgraphs by basic block.
    /// Edges are labeled with the register name.
    pub fn to_dot(&self, cfg: &ControlFlowGraph) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "digraph \"DDG: {}\" {{\n",
            self.cfg_name.replace('"', "\\\"")
        ));
        out.push_str("  rankdir=TB;\n");
        out.push_str(
            "  node [shape=box, fontname=\"monospace\", fontsize=10, margin=\"0.2,0.1\"];\n",
        );
        out.push_str("  edge [fontname=\"monospace\", fontsize=9];\n");

        // One subgraph per basic block
        for block in &cfg.blocks {
            let mut instr_idx = 0usize;
            let block_instrs: Vec<(InstrId, &str)> = block
                .statements
                .iter()
                .filter_map(|stmt| {
                    if matches!(stmt, FunctionStatement::Instruction { .. }) {
                        let id = (block.id, instr_idx);
                        instr_idx += 1;
                        self.nodes.get(&id).map(|txt| (id, txt.as_str()))
                    } else {
                        None
                    }
                })
                .collect();

            if block_instrs.is_empty() {
                continue;
            }

            let label = block
                .label
                .as_deref()
                .map(|l| format!("BB{} ({})", block.id, l))
                .unwrap_or_else(|| format!("BB{}", block.id));

            out.push_str(&format!("  subgraph cluster_BB{} {{\n", block.id));
            out.push_str(&format!(
                "    label=\"{}\";\n",
                label.replace('"', "\\\"")
            ));
            out.push_str(
                "    style=filled; fillcolor=\"#1e1e2e\"; color=\"#585b70\"; fontcolor=\"#cdd6f4\";\n",
            );

            for (id, ptx) in &block_instrs {
                let (block_id, idx) = id;
                let node_id = format!("I{}_{}", block_id, idx);
                let ptx_esc = dot_html_escape(ptx);

                let defs = self
                    .defs
                    .get(id)
                    .map(|v| v.join(", "))
                    .unwrap_or_default();
                let def_row = if defs.is_empty() {
                    String::new()
                } else {
                    format!(
                        "<TR><TD ALIGN=\"LEFT\"><FONT COLOR=\"#a6e3a1\">def: {}</FONT></TD></TR>",
                        dot_html_escape(&defs)
                    )
                };

                let label_html = format!(
                    "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\">\
                     <TR><TD ALIGN=\"LEFT\"><FONT COLOR=\"#cdd6f4\">{ptx_esc}</FONT></TD></TR>\
                     {def_row}\
                     </TABLE>"
                );
                out.push_str(&format!(
                    "    {node_id} [label=<{label_html}>, style=filled, fillcolor=\"#313244\", color=\"#585b70\"];\n"
                ));
            }

            out.push_str("  }\n");
        }

        // Edges: solid intra-block, dashed cross-block
        for edge in &self.edges {
            let (fb, fi) = edge.from;
            let (tb, ti) = edge.to;
            let from_id = format!("I{}_{}", fb, fi);
            let to_id = format!("I{}_{}", tb, ti);
            let reg_esc = dot_html_escape(&edge.register);
            let style = if edge.cross_block { "dashed" } else { "solid" };
            out.push_str(&format!(
                "  {from_id} -> {to_id} [label=\"{reg_esc}\", color=\"#89b4fa\", fontcolor=\"#89b4fa\", style=\"{style}\"];\n"
            ));
        }

        out.push_str("}\n");
        out
    }
}

// ---------------------------------------------------------------------------
// HTML export
// ---------------------------------------------------------------------------

impl DataDependencyGraph {
    /// Render a self-contained HTML page visualising the DDG with Mermaid.js.
    pub fn to_html(&self, cfg: &ControlFlowGraph) -> String {
        let mermaid = self.to_mermaid(cfg);
        let mermaid_escaped = mermaid
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DDG – {func}</title>
<style>
  body {{
    margin: 0;
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', system-ui, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  header {{
    flex-shrink: 0;
    padding: 12px 24px;
    background: #181825;
    border-bottom: 1px solid #313244;
    font-size: 1.1em;
  }}
  header code {{ color: #f9e2af; }}
  .source-file {{ color: #89b4fa; }}
  #graph-container {{
    flex: 1;
    overflow: hidden;
    position: relative;
  }}
  #graph-container svg {{
    width: 100%;
    height: 100%;
  }}
  /* hide the mermaid pre once rendered */
  .mermaid {{ display: contents; }}
</style>
</head>
<body>
<header>DDG for <code>{func}</code> <span class="source-file">(source: {file})</span></header>
<div id="graph-container">
  <pre class="mermaid">
{mermaid}
  </pre>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3/dist/svg-pan-zoom.min.js"></script>
<script>
  mermaid.initialize({{
    startOnLoad: false,
    theme: 'dark',
    flowchart: {{
      useMaxWidth: false,
      htmlLabels: true,
      curve: 'basis'
    }}
  }});

  mermaid.run({{ querySelector: '.mermaid' }}).then(() => {{
    const svg = document.querySelector('#graph-container svg');
    if (!svg) return;
    svg.style.width  = '100%';
    svg.style.height = '100%';
    svgPanZoom(svg, {{
      zoomEnabled: true,
      controlIconsEnabled: true,
      fit: true,
      center: true,
      minZoom: 0.02,
      maxZoom: 20,
    }});
  }});
</script>
</body>
</html>"#,
            func = cfg.function_name.replace('"', "&quot;"),
            file = cfg.source_file.replace('"', "&quot;"),
            mermaid = mermaid_escaped,
        )
    }
}

// ---------------------------------------------------------------------------
// Mermaid export
// ---------------------------------------------------------------------------

/// Escape a string for use inside a Mermaid `["..."]` node label.
/// Mermaid with htmlLabels=true interprets HTML tags, so we must escape
/// raw `<`/`>` that aren't intentional tags, and `"` always.
/// `%` is safe as-is; `[`/`]` need HTML entities to avoid confusing the parser.
fn mermaid_label_escape(s: &str) -> String {
    s.replace('"', "#quot;")
        .replace('[', "#91;")
        .replace(']', "#93;")
}

/// Escape a register name for a Mermaid edge label `-- reg -->`.
/// Strip the leading `%` so Mermaid doesn't choke on it in label position.
fn mermaid_edge_label(reg: &str) -> &str {
    reg.strip_prefix('%').unwrap_or(reg)
}

impl DataDependencyGraph {
    /// Render the DDG as a Mermaid flowchart string.
    ///
    /// Uses LR direction and subgraphs per basic block so the layout is readable.
    pub fn to_mermaid(&self, cfg: &ControlFlowGraph) -> String {
        let mut out = String::new();
        out.push_str("graph LR\n");

        for block in &cfg.blocks {
            let mut instr_idx = 0usize;
            let mut has_instrs = false;

            // Collect instructions first to know if the block is non-empty
            let instrs: Vec<(InstrId, String)> = block
                .statements
                .iter()
                .filter_map(|stmt| {
                    if !matches!(stmt, FunctionStatement::Instruction { .. }) {
                        return None;
                    }
                    let id = (block.id, instr_idx);
                    instr_idx += 1;
                    self.nodes.get(&id).map(|ptx| (id, ptx.clone()))
                })
                .collect();

            if instrs.is_empty() {
                continue;
            }

            let block_label = block
                .label
                .as_deref()
                .map(|l| format!("BB{} {}", block.id, mermaid_label_escape(l)))
                .unwrap_or_else(|| format!("BB{}", block.id));

            out.push_str(&format!("  subgraph SG{}[\"{block_label}\"]\n", block.id));
            out.push_str("    direction LR\n");

            for (id, ptx) in &instrs {
                has_instrs = true;
                let node_id = format!("I{}_{}", id.0, id.1);
                let ptx_esc = mermaid_label_escape(ptx);

                let defs = self.defs.get(id).map(|v| v.as_slice()).unwrap_or(&[]);
                let uses = self.uses.get(id).map(|v| v.as_slice()).unwrap_or(&[]);

                let def_line = if defs.is_empty() {
                    String::new()
                } else {
                    let regs = defs.iter().map(|r| mermaid_label_escape(r)).collect::<Vec<_>>().join(", ");
                    format!("<br/><b>def</b>: {regs}")
                };
                let use_line = if uses.is_empty() {
                    String::new()
                } else {
                    let regs = uses.iter().map(|r| mermaid_label_escape(r)).collect::<Vec<_>>().join(", ");
                    format!("<br/><b>use</b>: {regs}")
                };

                out.push_str(&format!(
                    "    {node_id}[\"{ptx_esc}{def_line}{use_line}\"]\n"
                ));
            }

            out.push_str("  end\n");
            let _ = has_instrs;
        }

        // Edges: solid for intra-block, dashed for cross-block
        for edge in &self.edges {
            let (fb, fi) = edge.from;
            let (tb, ti) = edge.to;
            let from_id = format!("I{}_{}", fb, fi);
            let to_id   = format!("I{}_{}", tb, ti);
            let label   = mermaid_edge_label(&edge.register);
            if edge.cross_block {
                out.push_str(&format!("  {from_id} -. {label} .-> {to_id}\n"));
            } else {
                out.push_str(&format!("  {from_id} -- {label} --> {to_id}\n"));
            }
        }

        out
    }
}
