use ptx_parser::PtxUnparser;
use ptx_parser::r#type::instruction::Inst;
use ptx_parser::r#type::meta::{MetaConstraint, MetaDirective, MetaTag};
use ptx_parser::r#type::{FunctionDim, FunctionStatement, Predicate};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Unique identifier for a basic block inside one function's CFG.
pub type BlockId = usize;

/// A single basic block: a maximal straight-line sequence of statements with
/// exactly one entry point (the first statement) and one exit point (the last).
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Unique id (index into `ControlFlowGraph::blocks`).
    pub id: BlockId,

    /// Optional label that starts this block.
    pub label: Option<String>,

    /// The statements that belong to this block.  Directives, instructions
    /// and nested blocks are all kept; only label statements that *start*
    /// a new block are split off.
    pub statements: Vec<FunctionStatement>,
}

/// The kind of terminator at the end of a basic block.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Terminator {
    /// Unconditional branch to a single target.
    Branch { target: String },
    /// Conditional (predicated) branch: may go to `target` or fall through.
    ConditionalBranch {
        predicate: Predicate,
        target: String,
        fallthrough: BlockId,
    },
    /// Indirect branch (`brx.idx`).  Targets are not statically known here
    /// so we record the instruction and treat it as opaque.
    IndirectBranch,
    /// Return from the current function.
    Return,
    /// `exit` – terminate the thread.
    Exit,
    /// The block falls through to the next one (no explicit branch).
    FallThrough { next: BlockId },
    /// Unreachable / empty tail (e.g. after an unconditional branch that is
    /// the last block).
    None,
}

/// An edge in the CFG.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub struct CfgEdge {
    pub from: BlockId,
    pub to: BlockId,
}

/// The complete control-flow graph for a single PTX function / entry kernel.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Path of the source PTX file this CFG was built from.
    pub source_file: String,

    /// Name of the function / entry point this CFG belongs to.
    pub function_name: String,

    /// Ordered list of basic blocks (index == `BlockId`).
    pub blocks: Vec<BasicBlock>,

    /// Forward edges  (from → set of to).
    pub successors: BTreeMap<BlockId, BTreeSet<BlockId>>,

    /// Backward edges (to → set of from).
    pub predecessors: BTreeMap<BlockId, BTreeSet<BlockId>>,

    /// Id of the entry block (always 0 when the function body is non-empty).
    pub entry: BlockId,

    /// Ids of blocks that end in `ret` or `exit` (or fall off the end).
    pub exits: Vec<BlockId>,

    /// `// @META` annotations found in the function body.
    pub meta: Vec<MetaDirective>,

    /// `.maxntid` header directive for this entry function, if present.
    ///
    /// For non-entry device functions this is always `None`.
    pub maxntid: Option<FunctionDim>,
}

// ---------------------------------------------------------------------------
// Pretty-printing helpers
// ---------------------------------------------------------------------------

impl fmt::Display for ControlFlowGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "CFG for `{}` (source: {}):",
            self.function_name, self.source_file
        )?;
        writeln!(f, "  entry: BB{}", self.entry)?;
        writeln!(
            f,
            "  exits: [{}]",
            self.exits
                .iter()
                .map(|id| format!("BB{id}"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(f)?;

        if !self.meta.is_empty() {
            writeln!(f, "  @META annotations ({}):", self.meta.len())?;
            for m in &self.meta {
                writeln!(f, "    {}", format_meta_line(m))?;
            }
            writeln!(f)?;
        }

        for block in &self.blocks {
            let label_str = block
                .label
                .as_deref()
                .map(|l| format!(" ({l})"))
                .unwrap_or_default();
            writeln!(
                f,
                "  BB{}{label_str}  [{} statement(s)]",
                block.id,
                block.statements.len()
            )?;

            if let Some(succs) = self.successors.get(&block.id) {
                let s: Vec<_> = succs.iter().map(|id| format!("BB{id}")).collect();
                writeln!(f, "    successors:   {}", s.join(", "))?;
            }
            if let Some(preds) = self.predecessors.get(&block.id) {
                let s: Vec<_> = preds.iter().map(|id| format!("BB{id}")).collect();
                writeln!(f, "    predecessors: {}", s.join(", "))?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Statement rendering helpers
// ---------------------------------------------------------------------------

/// Produce a short human-readable tag for an instruction variant,
/// e.g. `MovType`, `BraUni`, `SetpCmpopFtzType`, etc.
pub fn format_inst_short(inst: &Inst) -> String {
    let dbg = format!("{inst:?}");
    if let Some(paren) = dbg.find('(') {
        dbg[..paren].to_string()
    } else {
        dbg
    }
}

/// Render a single `FunctionStatement` as a one-line PTX string using the
/// parser's built-in unparser.  Newlines and leading/trailing whitespace are
/// stripped so the result is safe for embedding in graph labels.
fn format_stmt_ptx(stmt: &FunctionStatement) -> String {
    let tokens = stmt.to_tokens_spaced();
    let raw: String = tokens.iter().map(|t| t.as_str()).collect();
    raw.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format a statement as `[Tag]  ptx_text` for instructions, or just the PTX
/// text for labels / directives / blocks.
///
/// Returns `(tag, ptx_text)` where `tag` is `Some` for instructions.
fn format_stmt_parts(stmt: &FunctionStatement) -> (Option<String>, String) {
    let ptx = format_stmt_ptx(stmt);
    match stmt {
        FunctionStatement::Instruction { instruction, .. } => {
            let tag: String = format_inst_short(&instruction.inst);
            (Some(tag), ptx)
        }
        _ => (None, ptx),
    }
}

/// Combine the tag and PTX text into a single display line.
fn format_stmt_line(stmt: &FunctionStatement) -> String {
    let (tag, ptx) = format_stmt_parts(stmt);
    match tag {
        Some(t) => format!("[{t}]  {ptx}"),
        None => ptx,
    }
}

// ---------------------------------------------------------------------------
// Meta annotation formatting helpers
// ---------------------------------------------------------------------------

/// Format a single constraint for display.
fn format_constraint(c: &MetaConstraint) -> String {
    match c {
        MetaConstraint::Range { lo, hi } => format!("range={lo}:{hi}"),
        MetaConstraint::Stride { value } => format!("stride={value}"),
        MetaConstraint::Multiple { value } => format!("multiple={value}"),
        MetaConstraint::Align { value } => format!("align={value}"),
        MetaConstraint::ReadOnly => "readonly".to_string(),
        MetaConstraint::WriteOnly => "writeonly".to_string(),
        MetaConstraint::ReadWrite => "readwrite".to_string(),
        MetaConstraint::NoAlias => "noalias".to_string(),
        MetaConstraint::Other(s) => s.clone(),
    }
}

/// Format a MetaTag as a human-readable line for display in the CFG.
fn format_meta_tag(tag: &MetaTag) -> String {
    match tag {
        MetaTag::BeginKernel { name } => format!("BEGIN_KERNEL {name}"),
        MetaTag::EndKernel { name } => format!("END_KERNEL {name}"),
        MetaTag::Param { index, name, param_type, role, constraints } => {
            let cs: Vec<String> = constraints.iter().map(format_constraint).collect();
            let cs_str = if cs.is_empty() { String::new() } else { format!(" {}", cs.join(" ")) };
            format!("PARAM {index} {name} {param_type} {role}{cs_str}")
        }
        MetaTag::Tile { dim, size } => format!("TILE {dim} {size}"),
        MetaTag::Launch { block_x, block_y, block_z, grid_expr } => {
            format!("LAUNCH {block_x} {block_y} {block_z} {grid_expr}")
        }
        MetaTag::SharedMem { name, elem_type, total_bytes } => {
            format!("SHARED_MEM {name} {elem_type} {total_bytes}")
        }
        MetaTag::Loop { label, min_iters, max_iters, is_unrolled } => {
            format!("LOOP {label} {min_iters}..{max_iters} unrolled={is_unrolled}")
        }
        MetaTag::Layout { name, order, dims } => format!("LAYOUT {name} {order} {dims}"),
        MetaTag::Assume { description } => format!("ASSUME {description}"),
        MetaTag::ConstTable { symbol_name, num_entries } => {
            format!("CONST_TABLE {symbol_name} ({num_entries})")
        }
        MetaTag::Const { symbol_name, fields } => format!("CONST {symbol_name} {fields}"),
        MetaTag::Custom { key, value } => format!("CUSTOM {key} {value}"),
        MetaTag::Version { version } => format!("VERSION {version}"),
        MetaTag::Kernel { name } => format!("KERNEL {name}"),
        MetaTag::Unknown { raw } => raw.clone(),
    }
}

/// Format a MetaDirective as a single display line.
pub fn format_meta_line(m: &MetaDirective) -> String {
    let version_prefix = match m.version {
        Some(v) => format!("@META:{v} "),
        None => "@META ".to_string(),
    };
    format!("{version_prefix}{}", format_meta_tag(&m.tag))
}

// ---------------------------------------------------------------------------
// Mermaid export
// ---------------------------------------------------------------------------

impl ControlFlowGraph {
    /// Render the CFG as a Mermaid flowchart string.
    pub fn to_mermaid(&self) -> String {
        let mut out = String::new();
        out.push_str("graph TD\n");

        // Render @META annotations as a special info node
        if !self.meta.is_empty() {
            let meta_lines: Vec<String> = self.meta.iter().map(format_meta_line).collect();
            let meta_body = meta_lines.join("\n");
            let escaped = meta_body.replace('"', "#quot;");
            out.push_str(&format!(
                "    META[\"<b>@META annotations</b><br/><pre>{}</pre>\"]\n",
                escaped
            ));
            out.push_str("    style META fill:#2d2a4a,stroke:#b4befe,stroke-width:2px,color:#cdd6f4\n");
            out.push_str(&format!("    META -.-> BB{}\n", self.entry));
        }

        for block in &self.blocks {
            let stmt_lines: Vec<String> = block
                .statements
                .iter()
                .map(|s| format_stmt_line(s))
                .collect();

            let body = stmt_lines.join("\n");

            // Escape quotes and wrap in quoted node label.
            let escaped = body.replace('"', "#quot;");

            let mut shape_open = "[\"";
            let mut shape_close = "\"]";
            if self.exits.contains(&block.id) {
                shape_open = "[[\"";
                shape_close = "\"]]";
            }

            out.push_str(&format!(
                "    BB{}{shape_open}<b>BB{}</b><br/><pre>{}</pre>{shape_close}\n",
                block.id, block.id, escaped
            ));
        }

        // Edges
        for (from, tos) in &self.successors {
            for to in tos {
                out.push_str(&format!("    BB{from} --> BB{to}\n"));
            }
        }

        // Style the entry block
        out.push_str(&format!(
            "    style BB{} stroke-width:3px,stroke:#d35400\n",
            self.entry
        ));

        // Style exit blocks
        for exit in &self.exits {
            out.push_str(&format!(
                "    style BB{exit} stroke-width:3px,stroke:#27ae60\n"
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// DOT export – handy for visualisation with Graphviz
// ---------------------------------------------------------------------------

/// Escape a string for use inside a Graphviz HTML label.
fn dot_html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl ControlFlowGraph {
    /// Render the CFG as a Graphviz DOT string.
    ///
    /// Uses HTML labels so that instruction tags are shown in a separate
    /// column from the PTX text.  Nodes have generous margins so the text
    /// is never clipped.
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "digraph \"{}\" {{\n",
            self.function_name.replace('"', "\\\"")
        ));
        out.push_str("  rankdir=TB;\n");
        out.push_str(
            "  node [shape=box, fontname=\"monospace\", fontsize=10, margin=\"0.3,0.2\"];\n",
        );

        // Render @META annotations as a separate info node
        if !self.meta.is_empty() {
            let mut table = String::new();
            table.push_str(
                "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\">",
            );
            table.push_str(
                "<TR><TD COLSPAN=\"2\" ALIGN=\"CENTER\"><B>@META annotations</B></TD></TR>",
            );
            table.push_str("<HR/>");
            for m in &self.meta {
                let line = dot_html_escape(&format_meta_line(m));
                table.push_str(&format!(
                    "<TR><TD ALIGN=\"LEFT\"><FONT COLOR=\"#b4befe\">{line}</FONT></TD></TR>"
                ));
            }
            table.push_str("</TABLE>");
            out.push_str(&format!(
                "  META [label=<{table}>, shape=note, style=filled, fillcolor=\"#2d2a4a\", fontcolor=\"#cdd6f4\"];\n"
            ));
            out.push_str(&format!("  META -> BB{} [style=dashed, color=\"#888888\"];\n", self.entry));
        }

        for block in &self.blocks {
            // Build an HTML table label so tag and PTX text align in columns.
            let mut table = String::new();
            table.push_str(
                "<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\">",
            );

            // Header row with block id.
            table.push_str(&format!(
                "<TR><TD COLSPAN=\"2\" ALIGN=\"CENTER\"><B>BB{}</B></TD></TR>",
                block.id
            ));
            // Separator.
            table.push_str("<HR/>");

            for stmt in &block.statements {
                let (tag, ptx) = format_stmt_parts(stmt);
                let ptx_escaped = dot_html_escape(&ptx);

                match tag {
                    Some(t) => {
                        let tag_escaped = dot_html_escape(&t);
                        table.push_str(&format!(
                            "<TR><TD ALIGN=\"LEFT\"><FONT COLOR=\"#888888\">[{tag_escaped}]</FONT></TD>\
                             <TD ALIGN=\"LEFT\"> {ptx_escaped}</TD></TR>"
                        ));
                    }
                    None => {
                        table.push_str(&format!(
                            "<TR><TD COLSPAN=\"2\" ALIGN=\"LEFT\">{ptx_escaped}</TD></TR>"
                        ));
                    }
                }
            }

            table.push_str("</TABLE>");

            let mut attrs = format!("label=<{table}>");

            if block.id == self.entry {
                attrs.push_str(", style=bold, penwidth=2");
            }
            if self.exits.contains(&block.id) {
                attrs.push_str(", peripheries=2");
            }

            out.push_str(&format!("  BB{} [{attrs}];\n", block.id));
        }

        for (from, tos) in &self.successors {
            for to in tos {
                out.push_str(&format!("  BB{from} -> BB{to};\n"));
            }
        }

        out.push_str("}\n");
        out
    }
}
