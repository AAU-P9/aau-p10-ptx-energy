#!/usr/bin/env python3

import runner
import artifact
import plotter

def main(): 
    run_config = runner.parse_args()
    runner.run_runner(run_config)
    
    artifact_config = artifact.parse_args()
    artifact.run_artifact(artifact_config)
    
    plot_config = plotter.parse_args()
    plotter.run_plotter(plot_config)
    
    