use super::cfgMerge::{CallSite, MergedCfg};
use super::common::{BasicBlock, BlockId, ControlFlowGraph};
use ptx_parser::PtxUnparser;
use ptx_parser::r#type::instruction::Inst;
use ptx_parser::r#type::{FunctionStatement, GeneralOperand, Operand, StatementDirective};
use std::collections::BTreeSet;

fn html_escape(input: &str) -> String {
  input
    .replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
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

fn statement_call_targets(stmt: &FunctionStatement, out: &mut Vec<String>) {
  match stmt {
    FunctionStatement::Instruction { instruction, .. } => {
      let target = match &instruction.inst {
        Inst::CallUni(call) => extract_symbol_name(&call.func),
        Inst::CallUni1(call) => extract_symbol_name(&call.func),
        Inst::CallUni2(call) => extract_symbol_name(&call.func),
        _ => None,
      };

      if let Some(target) = target {
        out.push(target);
      }
    }
    FunctionStatement::Directive {
      directive: StatementDirective::CallTargets { directive, .. },
      ..
    } => {
      for target in &directive.targets {
        out.push(target.val.clone());
      }
    }
    FunctionStatement::Block { statements, .. } => {
      for nested in statements {
        statement_call_targets(nested, out);
      }
    }
    _ => {}
  }
}

fn stmt_ptx(stmt: &FunctionStatement) -> String {
  let tokens = stmt.to_tokens_spaced();
  let raw: String = tokens.iter().map(|token| token.as_str()).collect();
  raw.lines()
    .map(|line| line.trim())
    .filter(|line| !line.is_empty())
    .collect::<Vec<_>>()
    .join(" ")
}

fn extract_call_target_from_text(text: &str) -> Option<String> {
  if let Some(idx) = text.find(".calltargets") {
    let rest = text[idx + ".calltargets".len()..].trim();
    let target = rest
      .split(|ch: char| ch == ';' || ch == ',' || ch.is_whitespace())
      .find(|part| !part.is_empty())?;
    return Some(target.to_string());
  }

  let idx = text.find("call.uni").or_else(|| text.find("call "))?;
  let mut rest = text[idx..]
    .split_once(' ')
    .map(|(_, tail)| tail.trim())
    .unwrap_or_default();

  if rest.starts_with('(') {
    rest = rest.split_once(") ,").map(|(_, tail)| tail).unwrap_or(rest);
    if rest == text[idx..].split_once(' ').map(|(_, tail)| tail.trim()).unwrap_or_default() {
      rest = rest.split_once(",").map(|(_, tail)| tail.trim()).unwrap_or(rest);
    }
  }

  let target = rest
    .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
    .find(|part| !part.is_empty() && *part != "(")?;
  Some(target.to_string())
}

fn block_call_targets(block: &BasicBlock) -> Vec<String> {
  let mut targets = Vec::new();
  for stmt in &block.statements {
    statement_call_targets(stmt, &mut targets);
    if targets.is_empty() {
      if let Some(target) = extract_call_target_from_text(&stmt_ptx(stmt)) {
        targets.push(target);
      }
    }
  }
  targets.sort();
  targets.dedup();
  targets
}

fn collect_call_blocks(cfg: &ControlFlowGraph) -> Vec<(BlockId, Vec<String>)> {
  cfg.blocks
    .iter()
    .filter_map(|block| {
      let targets = block_call_targets(block);
      if targets.is_empty() {
        None
      } else {
        Some((block.id, targets))
      }
    })
    .collect()
}

fn detect_call_blocks_from_mermaid(mermaid: &str) -> Vec<(BlockId, Vec<String>)> {
  let mut current_block_id = None;
  let mut detected = BTreeSet::new();

  for line in mermaid.lines() {
    let trimmed = line.trim_start();

    if trimmed.starts_with("BB") {
      let block_id_text = trimmed[2..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
      current_block_id = block_id_text.parse::<BlockId>().ok();
    }

    if (line.contains("call.uni") || line.contains(".calltargets")) && current_block_id.is_some() {
      detected.insert(current_block_id.unwrap());
    }

    if trimmed.ends_with("]]") || trimmed.ends_with("]") {
      current_block_id = None;
    }
  }

  detected
    .into_iter()
    .map(|block_id| (block_id, vec!["function call".to_string()]))
    .collect()
}

fn block_class(block_id: BlockId, call_sites: &[CallSite], call_blocks: &[(BlockId, Vec<String>)]) -> &'static str {
  for call_site in call_sites {
    if block_id == call_site.call_block {
      return "blockCallsite";
    }

    if call_site.callee_return_blocks.contains(&block_id) {
      return "blockReturn";
    }

    if call_site.inlined_blocks.contains(&block_id) {
      return "blockInlined";
    }
  }

  if call_blocks.iter().any(|(id, _)| *id == block_id) {
    return "blockCallsite";
  }

  "blockNormal"
}

fn block_classes(block_id: BlockId, call_sites: &[CallSite], call_blocks: &[(BlockId, Vec<String>)]) -> Vec<&'static str> {
  let mut classes = Vec::new();

  if call_blocks.iter().any(|(id, _)| *id == block_id) {
    classes.push("blockCallsite");
  }

  for call_site in call_sites {
    if block_id == call_site.call_block {
      if !classes.contains(&"blockCallsite") {
        classes.push("blockCallsite");
      }
    }

    if block_id == call_site.callee_entry || call_site.inlined_blocks.contains(&block_id) {
      classes.push("blockInlined");
    }

    if call_site.callee_return_blocks.contains(&block_id) {
      classes.push("blockReturn");
    }
  }

  if classes.is_empty() {
    classes.push(block_class(block_id, call_sites, call_blocks));
  }

  classes
}

fn mermaid_class_lines(call_sites: &[CallSite], call_blocks: &[(BlockId, Vec<String>)]) -> String {
  if call_sites.is_empty() && call_blocks.is_empty() {
    return String::new();
  }

  let mut lines = String::new();
  lines.push_str("    classDef blockNormal fill:#313244,stroke:#6c7086,stroke-width:1px,color:#cdd6f4;\n");
  lines.push_str("    classDef blockCallsite fill:#3f2f19,stroke:#f9a826,stroke-width:3px,color:#f8f5ef;\n");
  lines.push_str("    classDef blockInlined fill:#1e3a34,stroke:#56c2a3,stroke-width:2px,color:#eefcf8;\n");
  lines.push_str("    classDef blockReturn fill:#2f2446,stroke:#9b8cff,stroke-width:2px,color:#f5f0ff;\n");

  let mut classified_blocks = BTreeSet::new();
  for (block_id, _) in call_blocks {
    classified_blocks.insert(*block_id);
  }
  for call_site in call_sites {
    classified_blocks.insert(call_site.call_block);
    for &block_id in &call_site.inlined_blocks {
      classified_blocks.insert(block_id);
    }
    for &block_id in &call_site.callee_return_blocks {
      classified_blocks.insert(block_id);
    }
  }

  for block_id in classified_blocks {
    let class_name = block_classes(block_id, call_sites, call_blocks).join(",");
    if class_name != "blockNormal" {
      lines.push_str(&format!("    class BB{block_id} {class_name};\n"));
    }
  }

  lines
}

fn call_sites_panel(call_sites: &[CallSite], call_blocks: &[(BlockId, Vec<String>)]) -> String {
  if call_sites.is_empty() && call_blocks.is_empty() {
    return String::new();
  }

  let mut rows = String::new();
  for call_site in call_sites {
    let caller_returns = if call_site.caller_return_blocks.is_empty() {
      "thread exit".to_string()
    } else {
      call_site
        .caller_return_blocks
        .iter()
        .map(|block_id| format!("BB{block_id}"))
        .collect::<Vec<_>>()
        .join(", ")
    };

    rows.push_str(&format!(
      "<tr><td>BB{}</td><td>{}</td><td>BB{}</td><td>{}</td></tr>",
      call_site.call_block,
      html_escape(&call_site.callee_name),
      call_site.callee_entry,
      caller_returns,
    ));
  }

  if call_sites.is_empty() {
    for (block_id, targets) in call_blocks {
      rows.push_str(&format!(
        "<tr><td>BB{}</td><td>{}</td><td>-</td><td>-</td></tr>",
        block_id,
        html_escape(&targets.join(", ")),
      ));
    }
  }

  format!(
    r#"<aside class="callsite-panel">
  <h2>Function Calls</h2>
  <table>
  <thead>
    <tr>
    <th>Call block</th>
    <th>Callee</th>
    <th>Entry</th>
    <th>Returns to</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
  </table>
</aside>"#,
    rows = rows
  )
}

/// Render a self-contained HTML page that visualises the CFG using Mermaid.js.
///
/// The page title and header include both the source file name and the
/// function name stored in the [`ControlFlowGraph`].
pub fn cfg_to_html(cfg: &ControlFlowGraph) -> String {
  render_cfg_html(cfg, &[])
}

pub fn merged_cfg_to_html(merged: &MergedCfg) -> String {
  render_cfg_html(&merged.cfg, &merged.call_sites)
}

/// Render a self-contained HTML page that visualises the CFG using Mermaid.js.
///
/// The page title and header include both the source file name and the
/// function name stored in the [`ControlFlowGraph`].
fn render_cfg_html(cfg: &ControlFlowGraph, call_sites: &[CallSite]) -> String {
  let mut mermaid = cfg.to_mermaid();
  let mut call_blocks = collect_call_blocks(cfg);
  if call_blocks.is_empty() {
    call_blocks = detect_call_blocks_from_mermaid(&mermaid);
  }
  mermaid.push_str(&mermaid_class_lines(call_sites, &call_blocks));

    // Escape the mermaid content for embedding inside HTML
    let mermaid_escaped = mermaid
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");

    let display_name = format!("{} – {}", cfg.source_file, cfg.function_name);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CFG – {title}</title>
<style>
  body {{
    margin: 0;
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
  }}
  header {{
    padding: 12px 24px;
    background: #181825;
    border-bottom: 1px solid #313244;
    font-size: 1.1em;
  }}
  main {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 360px;
    min-height: calc(100vh - 56px);
  }}
  header code {{
    color: #f9e2af;
  }}
  .source-file {{
    color: #89b4fa;
  }}
  #graph-container {{
    overflow: auto;
    display: flex;
    justify-content: center;
    padding: 24px;
  }}
  .callsite-panel {{
    border-left: 1px solid #313244;
    background: #181825;
    padding: 20px;
    overflow: auto;
  }}
  .callsite-panel h2 {{
    margin-top: 0;
    font-size: 1rem;
    color: #f9e2af;
  }}
  .callsite-panel table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }}
  .callsite-panel th,
  .callsite-panel td {{
    text-align: left;
    padding: 8px 0;
    border-bottom: 1px solid #313244;
    vertical-align: top;
  }}
  .mermaid {{
    text-align: center;
  }}
  /* mermaid node text */
  .mermaid pre {{
    text-align: left;
    margin: 0;
    font-size: 11px;
    line-height: 1.4;
  }}
  @media (max-width: 1080px) {{
    main {{
      grid-template-columns: 1fr;
    }}
    .callsite-panel {{
      border-left: 0;
      border-top: 1px solid #313244;
    }}
  }}
</style>
</head>
<body>
<header>CFG for <code>{func}</code> <span class="source-file">(source: {file})</span></header>
<main>
<div id="graph-container">
  <pre class="mermaid">
{mermaid}
  </pre>
</div>
{callsite_panel}
</main>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'dark',
    flowchart: {{
      useMaxWidth: false,
      htmlLabels: true,
      curve: 'basis'
    }}
  }});
</script>
</body>
</html>"#,
        title = display_name.replace('"', "&quot;"),
        func = cfg.function_name.replace('"', "&quot;"),
        file = cfg.source_file.replace('"', "&quot;"),
        mermaid = mermaid_escaped,
        callsite_panel = call_sites_panel(call_sites, &call_blocks),
    )
}
