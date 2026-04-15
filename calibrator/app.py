from pathlib import Path
from cubindings import execute_program
from analyser import run_analyser

gridDim = 64
blockDim = 1024

weights_path = Path("/home/rasmus/aau-p10-ptx-energy/linear-model/weights.csv")
vector_add_path = Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/matrix_mult/src") 

result = execute_program(path=vector_add_path, program_name="main.cu", debug=True)
response = run_analyser(result.path, weights_path, prediction=False, debug=True, program_name="main.cu")
print(list(response.analyser_result.instruction_occurrences.keys()))