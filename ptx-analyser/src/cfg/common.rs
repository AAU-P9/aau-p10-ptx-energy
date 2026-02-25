use ptx_parser::PtxUnparser;
use ptx_parser::r#type::instruction::Inst;
use ptx_parser::r#type::{FunctionStatement, Predicate};
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
fn format_inst_short(inst: &Inst) -> String {
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
            let tag = format_inst_short(&instruction.inst);
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
// Mermaid export
// ---------------------------------------------------------------------------

impl ControlFlowGraph {
    /// Render the CFG as a Mermaid flowchart string.
    pub fn to_mermaid(&self) -> String {
        let mut out = String::new();
        out.push_str("graph TD\n");

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
