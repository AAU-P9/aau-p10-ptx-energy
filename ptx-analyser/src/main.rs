use ptx_parser::parse_ptx;
use std::fs;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;
use serde::{Deserialize, Serialize};

use clap::{Parser, Subcommand};

mod cfg;
pub mod gpu_context;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Int,
    Float,
    UnsignedInt,
    SignedInt,
    Int64,
    Int32,
    Int16,
    Int8,
    Int4,
    Unknown, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub r#type: ParameterType,
    pub size: usize,
    pub value: Option<serde_json::Value>, // replaces void*
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelParameters {
    pub grid_dim: Dim3,
    pub block_dim: Dim3,
    pub parameters: Vec<Parameter>,
}

#[derive(Parser)]
#[command(name = "ptx-parser", about = "Utilities for parsing PTX assembly")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a PTX source file as a complete module.
    ParseFile {
        /// Path to the PTX source file to parse.
        input_file: PathBuf,
    },

    AnalyzeCfg {
        /// Path to the PTX source file to parse.
        input_file: PathBuf,

        /// Kernel parameters as a JSON string
        #[arg(long, conflicts_with = "kernel_params_file")]
        kernel_params: Option<String>,

        /// Path to JSON file containing kernel parameters
        #[arg(long, conflicts_with = "kernel_params")]
        kernel_params_file: Option<PathBuf>,

        #[arg(long)]
        /// Optional path to write the analysis result as JSON.
        output_json_path: Option<PathBuf>,

        #[arg(long)]
        /// Kernel name to include in the JSON output.
        kernel_name: Option<String>,

        #[arg(long)]
        /// Power consumption to include in the JSON output.
        power_consumption_joules: Option<f64>,
    },

    BuildCfg {
        /// Path to the PTX source file to parse.
        input_file: PathBuf,

        /// Optional path to write a Graphviz DOT file for each function.
        /// When provided, files are written as `<dot_output>.<function_index>.dot`.
        #[arg(long)]
        dot_output: Option<PathBuf>,

        /// Optional path to write a self-contained HTML visualisation for each function.
        /// When provided, files are written as `<html_output>.<function_index>.html`.
        #[arg(long)]
        html_output: Option<PathBuf>,

        /// Optional path to write an SVG rendering of the CFG for each function.
        /// Requires Graphviz `dot` to be installed.
        /// When provided, files are written as `<svg_output>.<function_index>.svg`.
        #[arg(long)]
        svg_output: Option<PathBuf>,
    },
}

/// Render a DOT string to SVG by invoking the Graphviz `dot` command.
/// Returns the SVG content as a string on success.
fn render_dot_to_svg(dot: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = ProcessCommand::new("dot")
        .args(["-Tsvg"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run `dot` (is Graphviz installed?): {}", e))
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin
                    .write_all(dot.as_bytes())
                    .map_err(|e| format!("Failed to write DOT to stdin: {}", e))?;
            }
            // Close stdin so dot can proceed
            drop(child.stdin.take());
            child
                .wait_with_output()
                .map_err(|e| format!("Failed to wait for `dot` process: {}", e))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Graphviz `dot` failed (exit {}):\n{}",
            output.status, stderr
        )
        .into());
    }

    Ok(String::from_utf8(output.stdout)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::ParseFile { input_file } => {
            let ptx_source = fs::read_to_string(&input_file)?;
            let module = parse_ptx(&ptx_source)?;

            // Serialize the full AST to JSON and save to file
            let json = serde_json::to_string_pretty(&module)?;
            std::fs::write("ast.json", &json)?;
            println!("Saved AST to ast.json ({} bytes)", json.len());
        }
        Command::AnalyzeCfg {
            input_file,
            kernel_params,
            kernel_params_file,
            output_json_path,
            kernel_name,
            power_consumption_joules,
        } => {
            let ptx_source = fs::read_to_string(&input_file)?;
            let module = parse_ptx(&ptx_source)?;
            
            let kernel_config: KernelParameters = if let Some(json) = kernel_params {
                serde_json::from_str(&json)
                    .map_err(|e| format!("Failed to parse kernel_params JSON: {}", e))?
            } else if let Some(path) = kernel_params_file {
                let contents = fs::read_to_string(&path)?;
                serde_json::from_str(&contents)
                    .map_err(|e| format!("Failed to parse JSON in {}: {}", path.display(), e))?
            } else {
                return Err("You must provide either --kernel-params or --kernel-params-file".into());
            };

            let file_name = input_file
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let cfg = cfg::cfg::build_cfg(&module, &file_name);

            println!("Analyzing CFGs with grid=({},{},{}), block=({},{},{}), parameters={:?}",
                kernel_config.grid_dim.x, kernel_config.grid_dim.y, kernel_config.grid_dim.z,
                kernel_config.block_dim.x, kernel_config.block_dim.y, kernel_config.block_dim.z,
                kernel_config.parameters);

            cfg::analyze_cfg(
                &cfg,
                kernel_config.grid_dim.x,
                kernel_config.grid_dim.y,
                kernel_config.grid_dim.z,
                kernel_config.block_dim.x,
                kernel_config.block_dim.y,
                kernel_config.block_dim.z,
                &kernel_config.parameters,
                &output_json_path,
                &kernel_name,
                &power_consumption_joules,
            );
        }
        Command::BuildCfg {
            input_file,
            dot_output,
            html_output,
            svg_output,
        } => {
            let ptx_source = fs::read_to_string(&input_file)?;
            let module = parse_ptx(&ptx_source)?;

            let file_name = input_file
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // --- new cfg rewrite ---
            let new_cfg = cfg::cfg::build_cfg(&module, &file_name);
            if let Some(base) = &html_output {
                let html_path = base.with_extension("new.html");
                let html = cfg::cfg::cfg_to_html(&new_cfg);
                fs::write(&html_path, &html)?;
                println!("new cfg -> HTML written to {}", html_path.display());
            }
            // -----------------------

            let cfgs = cfg::build_cfgs(&module, &file_name);

            if cfgs.is_empty() {
                println!("No functions with bodies found in the module.");
                return Ok(());
            }

            let root_name = cfgs
                .iter()
                .find(|cfg| cfg.is_entry)
                .map(|cfg| cfg.function_name.as_str())
                .unwrap_or_else(|| cfgs[0].function_name.as_str());

            if let Some(merged) = cfg::merge_cfgs(&cfgs, root_name) {
                let graph = &merged.cfg;
                println!("{}", graph);

                if let Some(base) = &dot_output {
                    let dot_path = base.with_extension("dot");
                    let dot = graph.to_dot();
                    fs::write(&dot_path, &dot)?;
                    println!(
                        "  -> DOT written to {} ({} bytes)\n",
                        dot_path.display(),
                        dot.len()
                    );

                    let ddg = cfg::ddg::build_ddg(graph);
                    let ddg_html_path = base.parent().unwrap_or(std::path::Path::new(".")).join("ddg.html");
                    let ddg_html = ddg.to_html(graph);
                    fs::write(&ddg_html_path, &ddg_html)?;
                    println!(
                        "  -> DDG HTML written to {} ({} bytes)\n",
                        ddg_html_path.display(),
                        ddg_html.len()
                    );
                }

                if let Some(base) = &html_output {
                    let html_path = base.with_extension("html");
                    let html = cfg::merged_cfg_to_html(&merged);
                    fs::write(&html_path, &html)?;
                    println!(
                        "  -> HTML written to {} ({} bytes)\n",
                        html_path.display(),
                        html.len()
                    );

                    let ddg = cfg::ddg::build_ddg(graph);
                    let ddg_html_path = base.parent().unwrap_or(std::path::Path::new(".")).join("ddg.html");
                    let ddg_html = ddg.to_html(graph);
                    fs::write(&ddg_html_path, &ddg_html)?;
                    println!(
                        "  -> DDG HTML written to {} ({} bytes)\n",
                        ddg_html_path.display(),
                        ddg_html.len()
                    );
                }

                if let Some(base) = &svg_output {
                    let svg_path = base.with_extension("svg");
                    let dot = graph.to_dot();
                    let svg = render_dot_to_svg(&dot)?;
                    fs::write(&svg_path, &svg)?;
                    println!(
                        "  -> SVG written to {} ({} bytes)\n",
                        svg_path.display(),
                        svg.len()
                    );

                    let ddg = cfg::ddg::build_ddg(graph);
                    let ddg_svg_path = base.with_extension("ddg.svg");
                    let ddg_dot = ddg.to_dot(graph);
                    let ddg_svg = render_dot_to_svg(&ddg_dot)?;
                    fs::write(&ddg_svg_path, &ddg_svg)?;
                    println!(
                        "  -> DDG SVG written to {} ({} bytes)\n",
                        ddg_svg_path.display(),
                        ddg_svg.len()
                    );
                }
            } else {
                for (i, graph) in cfgs.iter().enumerate() {
                    println!("{}", graph);

                    // Optionally write DOT files.
                    if let Some(base) = &dot_output {
                        let dot_path = if cfgs.len() == 1 {
                            base.with_extension("dot")
                        } else {
                            let stem = base
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            base.with_file_name(format!("{stem}.{i}.dot"))
                        };

                        let dot = graph.to_dot();
                        fs::write(&dot_path, &dot)?;
                        println!(
                            "  -> DOT written to {} ({} bytes)\n",
                            dot_path.display(),
                            dot.len()
                        );

                        let ddg = cfg::ddg::build_ddg(graph);
                        let ddg_dot_path = if cfgs.len() == 1 {
                            base.with_extension("ddg.dot")
                        } else {
                            let stem = base
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            base.with_file_name(format!("{stem}.{i}.ddg.dot"))
                        };
                        let ddg_dot = ddg.to_dot(graph);
                        fs::write(&ddg_dot_path, &ddg_dot)?;
                        println!(
                            "  -> DDG DOT written to {} ({} bytes)\n",
                            ddg_dot_path.display(),
                            ddg_dot.len()
                        );
                    }

                    // Optionally write HTML files.
                    if let Some(base) = &html_output {
                        let html_path = if cfgs.len() == 1 {
                            base.with_extension("html")
                        } else {
                            let stem = base
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            base.with_file_name(format!("{stem}.{i}.html"))
                        };

                        let html = cfg::cfg_to_html(graph);
                        fs::write(&html_path, &html)?;
                        println!(
                            "  -> HTML written to {} ({} bytes)\n",
                            html_path.display(),
                            html.len()
                        );

                        let ddg = cfg::ddg::build_ddg(graph);
                        let ddg_dot_path = if cfgs.len() == 1 {
                            base.with_extension("ddg.dot")
                        } else {
                            let stem = base
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            base.with_file_name(format!("{stem}.{i}.ddg.dot"))
                        };
                        let ddg_dot = ddg.to_dot(graph);
                        fs::write(&ddg_dot_path, &ddg_dot)?;
                        println!(
                            "  -> DDG DOT written to {} ({} bytes)\n",
                            ddg_dot_path.display(),
                            ddg_dot.len()
                        );
                    }

                    // Optionally render SVG via Graphviz dot.
                    if let Some(base) = &svg_output {
                        let svg_path = if cfgs.len() == 1 {
                            base.with_extension("svg")
                        } else {
                            let stem = base
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            base.with_file_name(format!("{stem}.{i}.svg"))
                        };

                        let dot = graph.to_dot();
                        let svg = render_dot_to_svg(&dot)?;
                        fs::write(&svg_path, &svg)?;
                        println!(
                            "  -> SVG written to {} ({} bytes)\n",
                            svg_path.display(),
                            svg.len()
                        );

                        let ddg = cfg::ddg::build_ddg(graph);
                        let ddg_svg_path = if cfgs.len() == 1 {
                            base.with_extension("ddg.svg")
                        } else {
                            let stem = base
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            base.with_file_name(format!("{stem}.{i}.ddg.svg"))
                        };
                        let ddg_dot = ddg.to_dot(graph);
                        let ddg_svg = render_dot_to_svg(&ddg_dot)?;
                        fs::write(&ddg_svg_path, &ddg_svg)?;
                        println!(
                            "  -> DDG SVG written to {} ({} bytes)\n",
                            ddg_svg_path.display(),
                            ddg_svg.len()
                        );
                    }
                }
            }
        }
    }

    Ok(())
}
