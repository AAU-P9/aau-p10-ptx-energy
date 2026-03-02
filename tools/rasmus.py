#!/usr/bin/env python3

import runner
import artifact

def main(): 
    run_config = runner.parse_args()
    runner.run_runner(run_config)
    
    artifact_config = artifact.parse_args()
    artifact.run_artifact(artifact_config)
    
    