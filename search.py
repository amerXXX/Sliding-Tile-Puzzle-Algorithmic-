
import time
import heapq
import math
from collections import deque

class Search:
    def __init__(self, goal_test, next_states, state=None, goal_puzzle=None):

        self.goal_test = goal_test
        self.next_states = next_states
        self.goal_puzzle = goal_puzzle

        if state is not None:
            self.state = state
        else:
            raise ValueError("Initial state must be provided.")

        # Algorithms dictionary mapping algorithm names to methods
        self.algorithms = {
            'bfs': self.bfs,
            'dfs': self.dfs,
            'dfs_visited': self.dfs_visited,
            'astar_misplaced': self.astar_misplaced,
            'astar_manhattan': self.astar_manhattan,
            'astar_euclidean': self.astar_euclidean,
            'dls': self.dls,
            'dfids': self.dfids,
            'ida_star': self.ida_star,
        }

        # To track parent of each state for path reconstruction
        self.parents = {}
        self.parents[tuple(state)] = None

        # Attributes to store statistics of the last search
        self.last_depth = None
        self.last_states = None
        self.execution_time = None
        self.max_concurrent_states = 0

    def set_state(self, state):

        self.state = state
        self.parents = {}
        self.parents[tuple(state)] = None
        self.last_depth = None
        self.last_states = None
        self.execution_time = None
        self.max_concurrent_states = 0

    def get_depth(self, state):

        depth = 0
        current = tuple(state)
        while self.parents.get(current) is not None:
            current = self.parents[current]
            depth += 1
        return depth

    def get_path(self, state):

        path = []
        current = tuple(state)
        while current is not None:
            path.append(list(current))
            current = self.parents.get(current)
        path.reverse()
        return path

    def search(self, algorithm: str, verbose=True, **kwargs) -> list:

        if not isinstance(algorithm, str):
            raise Exception("type(algorithm) must be string.")
        if algorithm not in self.algorithms:
            raise Exception(f"No algorithm named {algorithm} found.")

        start_time = time.time()
        solution = self.algorithms[algorithm](verbose=verbose, **kwargs)
        end_time = time.time()
        self.execution_time = end_time - start_time

        if verbose:
            print(f"Time taken: {self.execution_time:.2f} seconds")

        return solution

    # --------------- Search Methods ---------------

    def bfs(self, verbose: bool = True) -> list:

        if verbose:
            print("************** Solving (BFS) *****************")
        queue = deque([self.state])
        visited = set()
        states = 0

        while queue:
            if verbose:
                print(f"\rQueue Size: {len(queue)} | States Explored: {states}", end='')
            current = queue.popleft()
            current_tuple = tuple(current)

            if current_tuple in visited:
                continue
            visited.add(current_tuple)
            states += 1

            if self.goal_test(current):
                if verbose:
                    print()
                self.last_depth = self.get_depth(current)
                self.last_states = states
                return self.get_path(current)

            for neighbor in self.next_states(current):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    queue.append(neighbor)
                    self.parents[neighbor_tuple] = current_tuple

            self.max_concurrent_states = max(self.max_concurrent_states, len(queue))

        raise Exception("Can't find Solution.")

    def dfs(self, verbose: bool = True, max_depth=1000) -> list:

        if verbose:
            print("************** Solving (DFS) *****************")
        stack = [self.state]
        visited = set()
        states = 0

        while stack:
            if verbose:
                print(f"\rStack Size: {len(stack)} | States Explored: {states}", end='')
            current = stack.pop()
            current_tuple = tuple(current)

            if current_tuple in visited:
                continue
            visited.add(current_tuple)
            states += 1

            if self.goal_test(current):
                if verbose:
                    print()
                self.last_depth = self.get_depth(current)
                self.last_states = states
                return self.get_path(current)

            if self.get_depth(current) < max_depth:
                for neighbor in reversed(self.next_states(current)):
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple not in visited:
                        stack.append(neighbor)
                        self.parents[neighbor_tuple] = current_tuple

            self.max_concurrent_states = max(self.max_concurrent_states, len(stack))

        raise Exception("Can't find Solution.")

    def dfs_visited(self, verbose: bool = True, max_depth=1000) -> list:

        if verbose:
            print("************** Solving (DFS_VISITED) *****************")
        stack = [self.state]
        visited = set()
        states = 0

        while stack:
            if verbose:
                print(f"\rStack Size: {len(stack)} | States Explored: {states}", end='')
            current = stack.pop()
            current_tuple = tuple(current)

            if current_tuple in visited:
                continue
            visited.add(current_tuple)
            states += 1

            if self.goal_test(current):
                if verbose:
                    print()
                self.last_depth = self.get_depth(current)
                self.last_states = states
                return self.get_path(current)

            if self.get_depth(current) < max_depth:
                for neighbor in reversed(self.next_states(current)):
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple not in visited:
                        stack.append(neighbor)
                        self.parents[neighbor_tuple] = current_tuple

            self.max_concurrent_states = max(self.max_concurrent_states, len(stack))

        raise Exception("Can't find Solution.")

    def dls(self, depth: int = 0, verbose: bool = True, get_states: bool = False) -> [list, int]:

        if verbose:
            print("************** Solving (DLS) *****************")
        stack = [(self.state, 0)]  # Each element is a tuple (state, current_depth)
        visited = set()
        states = 0

        while stack:
            if verbose:
                print(f"\rStack Size: {len(stack)} | States Explored: {states}", end='')
            current, current_depth = stack.pop()
            current_tuple = tuple(current)

            if current_tuple in visited:
                continue
            visited.add(current_tuple)
            states += 1

            if self.goal_test(current):
                if verbose:
                    print()
                self.last_depth = current_depth
                self.last_states = states
                return self.get_path(current)

            if current_depth < depth:
                for neighbor in reversed(self.next_states(current)):
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple not in visited:
                        stack.append((neighbor, current_depth + 1))
                        self.parents[neighbor_tuple] = current_tuple

            self.max_concurrent_states = max(self.max_concurrent_states, len(stack))

        if get_states:
            return states
        raise Exception("Can't find Solution in the specified depth. Try increasing depth.")

    def dfids(self, verbose: bool = True) -> list:

        if verbose:
            print("************** Solving (DFIDS) *****************")
        depth_limit = 0
        total_states = 0

        while True:
            if verbose:
                print(f"\rIteration: {depth_limit} | Total States Explored: {total_states}", end='')
            try:
                result = self.dls(depth=depth_limit, verbose=False, get_states=True)
                total_states += result
                # Attempt to solve with current depth limit
                solution = self.dls(depth=depth_limit, verbose=False, get_states=False)
                if solution:
                    self.last_depth = depth_limit
                    self.last_states = total_states
                    return solution
            except Exception:
                # No solution found at current depth, increase depth limit
                depth_limit += 1

    # Heuristic Functions
    def misplaced_tiles(self, state):

        return sum(1 for i, tile in enumerate(state) if tile != 0 and tile != self.goal_puzzle[i])

    def manhattan_distance(self, state):

        distance = 0
        size = int(math.sqrt(len(state)))
        for i, tile in enumerate(state):
            if tile != 0:
                goal_index = self.goal_puzzle.index(tile)
                current_row, current_col = divmod(i, size)
                goal_row, goal_col = divmod(goal_index, size)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance

    def euclidean_distance(self, state):

        distance = 0.0
        size = int(math.sqrt(len(state)))
        for i, tile in enumerate(state):
            if tile != 0:
                goal_index = self.goal_puzzle.index(tile)
                current_row, current_col = divmod(i, size)
                goal_row, goal_col = divmod(goal_index, size)
                distance += math.sqrt((current_row - goal_row) ** 2 + (current_col - goal_col) ** 2)
        return distance

    # A* Search Methods
    def astar(self, heuristic, verbose=True) -> list:

        if verbose:
            print(f"************** Solving (A*) using {heuristic.__name__} *****************")
        open_set = []
        heapq.heappush(open_set, (heuristic(self.state), 0, self.state))
        came_from = {}
        g_score = {tuple(self.state): 0}
        states = 1
        self.max_concurrent_states = max(self.max_concurrent_states, len(open_set))

        visited = set()

        while open_set:
            if verbose:
                print(f"\rOpen Set Size: {len(open_set)} | States Explored: {states}", end='')
            f_current, g_current, current = heapq.heappop(open_set)
            current_tuple = tuple(current)

            if current_tuple in visited:
                continue

            if self.goal_test(current):
                if verbose:
                    print()
                self.last_depth = g_current
                self.last_states = states
                return self.get_path(current)

            visited.add(current_tuple)

            for neighbor in self.next_states(current):
                neighbor_tuple = tuple(neighbor)
                tentative_g_score = g_current + 1  # Assuming cost between states is 1

                if neighbor_tuple in visited and tentative_g_score >= g_score.get(neighbor_tuple, float('inf')):
                    continue

                if tentative_g_score < g_score.get(neighbor_tuple, float('inf')):
                    came_from[neighbor_tuple] = current_tuple
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                    states += 1
                    self.parents[neighbor_tuple] = current_tuple

            self.max_concurrent_states = max(self.max_concurrent_states, len(open_set))

        raise Exception("Can't find Solution.")

    def reconstruct_path(self, came_from, current):

        path = [current]
        current_tuple = tuple(current)
        while current_tuple in came_from:
            current = came_from[current_tuple]
            path.append(current)
            current_tuple = tuple(current)
        path.reverse()
        return path

    def astar_misplaced(self, verbose=True) -> list:

        return self.astar(self.misplaced_tiles, verbose)

    def astar_manhattan(self, verbose=True) -> list:

        return self.astar(self.manhattan_distance, verbose)

    def astar_euclidean(self, verbose=True) -> list:

        return self.astar(self.euclidean_distance, verbose)

    # Depth Limited Search (DLS)
    def dls(self, depth: int = 0, verbose: bool = True, get_states: bool = False) -> [list, int]:

        if verbose:
            print("************** Solving (DLS) *****************")
        stack = [(self.state, 0)]  # Each element is a tuple (state, current_depth)
        visited = set()
        states = 0

        while stack:
            if verbose:
                print(f"\rStack Size: {len(stack)} | States Explored: {states}", end='')
            current, current_depth = stack.pop()
            current_tuple = tuple(current)

            if current_tuple in visited:
                continue
            visited.add(current_tuple)
            states += 1

            if self.goal_test(current):
                if verbose:
                    print()
                self.last_depth = current_depth
                self.last_states = states
                return self.get_path(current)

            if current_depth < depth:
                for neighbor in reversed(self.next_states(current)):
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple not in visited:
                        stack.append((neighbor, current_depth + 1))
                        self.parents[neighbor_tuple] = current_tuple

            self.max_concurrent_states = max(self.max_concurrent_states, len(stack))

        if get_states:
            return states
        raise Exception("Can't find Solution in the specified depth. Try increasing depth.")

    # Depth First Iterative Deepening Search (DFIDS)
    def dfids(self, verbose: bool = True) -> list:

        if verbose:
            print("************** Solving (DFIDS) *****************")
        depth_limit = 0
        total_states = 0

        while True:
            if verbose:
                print(f"\rIteration: {depth_limit} | Total States Explored: {total_states}", end='')
            try:
                result = self.dls(depth=depth_limit, verbose=False, get_states=True)
                total_states += result
                # Attempt to solve with current depth limit
                solution = self.dls(depth=depth_limit, verbose=False, get_states=False)
                if solution:
                    self.last_depth = depth_limit
                    self.last_states = total_states
                    return solution
            except Exception:
                # No solution found at current depth, increase depth limit
                depth_limit += 1

    # Iterative Deepening A* (IDA*)
    def ida_star(self, heuristic, verbose=True) -> list:

        if verbose:
            print(f"************** Solving (IDA*) using {heuristic.__name__} *****************")
        self.max_concurrent_states = 0  # Reset for IDA*
        threshold = heuristic(self.state)
        path = [self.state]
        g = 0

        while True:
            if verbose:
                print(f"Initial threshold: {threshold}")
            temp = self._ida_star_recursive(path, g, threshold, heuristic, verbose)
            if isinstance(temp, list):
                return temp
            if temp == float('inf'):
                raise Exception("Can't find Solution.")
            threshold = temp

    def _ida_star_recursive(self, path, g, threshold, heuristic, verbose):

        current_state = path[-1]
        f = g + heuristic(current_state)
        if f > threshold:
            return f
        if self.goal_test(current_state):
            return path
        min_threshold = float('inf')
        for neighbor in self.next_states(current_state):
            if neighbor in path:
                continue  # Avoid cycles
            path.append(neighbor)
            self.max_concurrent_states = max(self.max_concurrent_states, len(path))
            temp = self._ida_star_recursive(path, g + 1, threshold, heuristic, verbose)
            if isinstance(temp, list):
                return temp
            if temp < min_threshold:
                min_threshold = temp
            path.pop()
        return min_threshold

    # --------------- End of Search Methods ---------------

