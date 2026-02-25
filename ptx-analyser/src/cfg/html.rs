use super::common::ControlFlowGraph;

/// Render a self-contained HTML page that visualises the CFG using Mermaid.js.
///
/// The page title and header include both the source file name and the
/// function name stored in the [`ControlFlowGraph`].
pub fn cfg_to_html(cfg: &ControlFlowGraph) -> String {
    let mermaid = cfg.to_mermaid();
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
    display: flex;
    flex-direction: column;
    height: 100vh;
  }}
  header {{
    padding: 12px 24px;
    background: #181825;
    border-bottom: 1px solid #313244;
    font-size: 1.1em;
  }}
  header code {{
    color: #f9e2af;
  }}
  .source-file {{
    color: #89b4fa;
  }}
  #graph-container {{
    flex: 1;
    overflow: auto;
    display: flex;
    justify-content: center;
    padding: 24px;
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
</style>
</head>
<body>
<header>CFG for <code>{func}</code> <span class="source-file">(source: {file})</span></header>
<div id="graph-container">
  <pre class="mermaid">
{mermaid}
  </pre>
</div>
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
    )
}
