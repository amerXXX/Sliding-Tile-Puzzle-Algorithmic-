# Sliding-Tile-Puzzle-Algorithm-
a Python implementation of the m‑puzzle (n × n sliding‑tile puzzle) solver using three search strategies: 
Breadth‑First Search (BFS)  
Depth‑First Search (DFS) 
DFS with cycle checking

# Results and Analysis

See the results/ directory for raw CSVs and results/summary.md for consolidated tables and commentary:

- BFS performs optimally on shallow puzzles but scales poorly: state‑space explosion limits .

- DFS often fails to find solutions without cycle prevention and is highly variable in runtime.

- DFS with cycle checking dramatically reduces redundant exploration but still struggles on larger boards.

# Extending the Project

Add heuristic best‑first or A* search

Implement graphical animation of solution steps

Parallelize benchmarking across multiple cores

# License

This project is released under the MIT License. Feel free to reuse and adapt.
