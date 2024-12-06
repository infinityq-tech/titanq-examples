import numpy as np

def parse_input(filename: str) -> tuple[np.ndarray, int]:
  """
  Read in a DIMACS instance from a given file path and return
  the corresponding adjacency matrix and number of vertices.

  Args:
    filename (str): The relative path of the DIMACS instance.

  Returns:
    np.ndarray: Adjacency matrix representing the instance provided.
    int: The number of vertices in the provided instance.
  """
  f = open(filename)
  for line in f:
    instr = line[0]
    if instr == 'p':
      _, _, N_str, _ = line.split()
      N = int(N_str)
      adj = np.zeros((N, N), dtype=np.float32)
    elif instr == 'e':
      _, a, b = line.split()
      adj[int(a) - 1, int(b) - 1] = 1
      adj[int(b) - 1, int(a) - 1] = 1
  f.close()
  return adj, N
