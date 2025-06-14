import torch
from typing import Optional, Tuple, List

# The same everywhere
# BOARD_SIZE = 19
# EMPTY = 0
# BLACK = -1
# WHITE = 1

# def opponent(player: int) -> int:
#     return BLACK if player == WHITE else WHITE
#
# def is_on_board(x: int, y: int) -> bool:
#     return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
#
# def get_neighbors(x: int, y: int) -> List[Tuple[int,int]]:
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     return [(x+dx, y+dy) for dx, dy in directions if is_on_board(x+dx, y+dy)]

class GoGame:
    def __init__(self, BOARD_SIZE):
        self.BOARD_SIZE = BOARD_SIZE
        self.EMPTY = 0
        self.BLACK = -1
        self.WHITE = 1
        self.komi = 7.5

        self.board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.int8)
        self.history: List[torch.Tensor] = []
        self.move_log: List[Tuple[int, Optional[int], Optional[int]]] = []
        self.current_player = self.BLACK
        self.pass_count = 0
        self.game_over = False
        self.last_move: Optional[Tuple[int,int]] = None
        self.captures = {self.BLACK: 0, self.WHITE: 0}

    def opponent(self, player: int) -> int:
        return self.BLACK if player == self.WHITE else self.WHITE

    def is_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if self.is_on_board(x + dx, y + dy)]

    def copy_board(self) -> torch.Tensor:
        return self.board.clone()

    def _flood_fill_group(
        self,
        bx: int,
        by: int,
        board_tensor: torch.Tensor,
        visited: Optional[set] = None
    ) -> set:
        """
        Flood‐fill from (bx,by) on board_tensor (19×19) to find all connected stones of that color.
        """
        color = board_tensor[bx, by].item()
        group = set()
        stack = [(bx, by)]
        visited = visited or set()
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited or board_tensor[cx, cy].item() != color:
                continue
            visited.add((cx, cy))
            group.add((cx, cy))
            for nx, ny in self.get_neighbors(cx, cy):
                if (nx, ny) not in visited:
                    stack.append((nx, ny))
        return group

    def _count_liberties(
        self,
        group: set,
        board_tensor: torch.Tensor
    ) -> int:
        """
        Count liberties of a given group of coordinates on board_tensor.
        """
        liberties = set()
        for x, y in group:
            for nx, ny in self.get_neighbors(x, y):
                if board_tensor[nx, ny].item() == self.EMPTY:
                    liberties.add((nx, ny))
        return len(liberties)

    def remove_dead_stones(
        self,
        player: int
    ):
        """
        Removes opponent groups with no liberties after the last placed stone (self.last_move),
        operating on self.board. Increments self.captures[player].
        """
        if self.last_move is None:
            return

        to_remove = []
        visited = set()
        x0, y0 = self.last_move
        # print(f"\nChecking for captures after move at ({x0}, {y0})")
        
        for nx, ny in self.get_neighbors(x0, y0):
            if (nx, ny) not in visited and self.board[nx, ny].item() == self.opponent(player):
                # print(f"Found opponent stone at ({nx}, {ny})")
                group = self._flood_fill_group(nx, ny, self.board, visited)
                liberties = self._count_liberties(group, self.board)
                # print(f"Group has {liberties} liberties")
                if liberties == 0:
                    # print(f"Group has no liberties, will be captured")
                    to_remove.extend(group)
                # else:
                #     # print(f"Group has liberties, not captured")

        if to_remove:
            # print(f"Removing {len(to_remove)} stones")
            for rx, ry in to_remove:
                self.board[rx, ry] = self.EMPTY
            self.captures[player] += len(to_remove)
            # print(f"Total captures for {'Black' if player == self.BLACK else 'White'}: {self.captures[player]}")
        # else:
        #     print("No captures this move")

    def is_suicide(
        self,
        x: int,
        y: int,
        player: int
    ) -> bool:
        """
        Returns True if placing `player` at (x,y) on a temporary board would be suicide.
        Correctly simulates on temp_board:
         1) Place player at (x,y).
         2) Remove any adjacent opponent groups with zero liberties on temp_board.
         3) Flood‐fill your newly placed group on temp_board and check its liberties.
        """
        # 1) Copy and place the stone
        temp_board = self.copy_board()
        temp_board[x, y] = player

        # 2) Remove any capturable adjacent opponent groups (using temp_board)
        visited = set()
        for nx, ny in self.get_neighbors(x, y):
            if temp_board[nx, ny].item() == self.opponent(player) and (nx, ny) not in visited:
                group = self._flood_fill_group(nx, ny, temp_board, visited)
                if self._count_liberties(group, temp_board) == 0:
                    for gx, gy in group:
                        temp_board[gx, gy] = self.EMPTY

        # 3) Now flood‐fill your own group containing (x,y) on temp_board
        group = self._flood_fill_group(x, y, temp_board, visited=None)
        return (self._count_liberties(group, temp_board) == 0)

    def is_legal(
        self,
        x: int,
        y: int
    ) -> bool:
        """
        Returns False if:
          - (x,y) is off‐board or not empty
          - move is suicide on temp_board
          - move violates Ko (i.e. temp_board after one move exactly equals history[-1])
        Otherwise True.
        """
        # 1) Off-board or occupied?
        if not self.is_on_board(x, y):
            return False
        if self.board[x, y].item() != self.EMPTY:
            return False

        # 2) Suicide?
        if self.is_suicide(x, y, self.current_player):
            return False

        # 3) Ko check: simulate one move on a backup board and see if it repeats history[-1]
        backup = self.copy_board()
        backup[x, y] = self.current_player

        # Remove any captures on backup
        visited = set()
        for nx, ny in self.get_neighbors(x, y):
            if backup[nx, ny].item() == self.opponent(self.current_player) and (nx, ny) not in visited:
                group = self._flood_fill_group(nx, ny, backup, visited)
                if self._count_liberties(group, backup) == 0:
                    for gx, gy in group:
                        backup[gx, gy] = self.EMPTY

        # If after this simulated move the board equals the previous position, forbid it
        if self.history and torch.equal(backup, self.history[-1]):
            return False

        return True

    def play_move(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None
    ) -> bool:
        """
        Play a move at (x,y). If x or y is None, that is a pass.
        Returns False if illegal or game over; otherwise True.
        """
        if self.game_over:
            return False

        # Passing
        if x is None or y is None:
            self.move_log.append((self.current_player, None, None))
            self.pass_count += 1
            if self.pass_count >= 2:
                self.game_over = True
            self.current_player = self.opponent(self.current_player)
            return True

        # Illegal?
        if not self.is_legal(x, y):
            return False

        # Place stone
        self.board[x, y] = self.current_player
        self.last_move = (x, y)

        # Remove dead opponent stones and count captures
        prev_captures = self.captures[self.current_player]
        self.remove_dead_stones(self.current_player)
        new_captures = self.captures[self.current_player]
        # if new_captures > prev_captures:
            # print(f"{'Black' if self.current_player == self.BLACK else 'White'} captured {new_captures - prev_captures} stones")

        # Append to history for Ko
        self.history.append(self.copy_board())

        # Log the move
        self.move_log.append((self.current_player, x, y))
        self.pass_count = 0
        self.current_player = self.opponent(self.current_player)
        return True

    def estimate_territory(self) -> dict:
        """
        Implements area scoring rules:
        1. Count all intersections enclosed by your stones (including empty points)
        2. Add captured stones
        3. Count empty points in moyo (large open areas) if bounded by your stones
        4. Count empty points inside opponent's isolated groups (if they don't form their own territory)
        
        Returns {'black_territory': int, 'white_territory': int}.
        """
        visited = set()
        territory = {self.BLACK: 0, self.WHITE: 0}
        
        def flood_fill_territory(x: int, y: int, owner: int) -> Tuple[set, set, bool]:
            """
            Flood fill from (x,y) to find all connected empty points and their borders.
            Returns (empty_points, border_stones, has_opponent_wall)
            """
            empty_points = set()
            border_stones = set()
            has_opponent_wall = False
            stack = [(x, y)]
            
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                    
                val = self.board[cx, cy].item()
                if val == self.EMPTY:
                    visited.add((cx, cy))
                    empty_points.add((cx, cy))
                    
                    # Check neighbors
                    for nx, ny in self.get_neighbors(cx, cy):
                        if (nx, ny) not in visited:
                            nval = self.board[nx, ny].item()
                            if nval == self.EMPTY:
                                stack.append((nx, ny))
                            else:
                                border_stones.add((nx, ny))
                                if nval == self.opponent(owner):
                                    has_opponent_wall = True
                elif val == owner:
                    border_stones.add((cx, cy))
                elif val == self.opponent(owner):
                    has_opponent_wall = True
                    
            return empty_points, border_stones, has_opponent_wall

        # First pass: identify all empty regions and their borders
        empty_regions = []
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if (i, j) not in visited and self.board[i, j].item() == self.EMPTY:
                    # Try both colors as potential owners
                    for owner in [self.BLACK, self.WHITE]:
                        empty_points, border_stones, has_opponent_wall = flood_fill_territory(i, j, owner)
                        if empty_points:  # If we found a new region
                            empty_regions.append((empty_points, border_stones, has_opponent_wall, owner))
                            break  # Only need to check one color per empty point

        # Second pass: score each region
        for empty_points, border_stones, has_opponent_wall, owner in empty_regions:
            # Count stones of each color in the border
            border_counts = {self.BLACK: 0, self.WHITE: 0}
            for x, y in border_stones:
                border_counts[self.board[x, y].item()] += 1
                
            # Determine territory ownership
            if not has_opponent_wall:  # No opponent stones touching the region
                territory[owner] += len(empty_points)
            # elif border_counts[owner] > border_counts[self.opponent(owner)]:
            #     # Your stones form a stronger wall around the region
            #     territory[owner] += len(empty_points)
            # elif border_counts[owner] == border_counts[self.opponent(owner)]:
            #     # Equal influence - split the territory
            #     territory[owner] += len(empty_points) // 2
            #     territory[self.opponent(owner)] += len(empty_points) // 2

        return {'black_territory': territory[self.BLACK],
                'white_territory': territory[self.WHITE]}

    def score(self) -> dict:
        """
        Full scoring = territory + captures:
          black_score = black_territory + black_captures
          white_score = white_territory + white_captures
        """
        terr = self.estimate_territory()
        black_ter = terr['black_territory']
        white_ter = terr['white_territory']
        return {
            'black_score': black_ter + self.captures[self.BLACK],
            'white_score': white_ter + self.captures[self.WHITE] + self.komi
        }

    def print_board(self):
        symbols = {self.BLACK: '●', self.WHITE: '○', self.EMPTY: '+'}
        print("    " + ' '.join(f'{i:2}' for i in range(self.BOARD_SIZE)))
        for i in range(self.BOARD_SIZE):
            row = f'{i:2} '
            for j in range(self.BOARD_SIZE):
                row += symbols[self.board[i, j].item()] + '  '
            print(row)

    def print_move_log(self):
        for idx, move in enumerate(self.move_log):
            player = 'B' if move[0] == self.BLACK else 'W'
            if move[1] is None:
                pos = "pass"
            else:
                pos = f"({move[1]}, {move[2]})"
            print(f'{idx+1:3}: {player} → {pos}')

    def clone(self) -> "GoGame":
        new = GoGame(self.BOARD_SIZE)
        new.board = self.board.clone()
        new.history = [h.clone() for h in self.history]
        new.move_log = list(self.move_log)
        new.current_player = self.current_player
        new.pass_count = self.pass_count
        new.game_over = self.game_over
        new.last_move = None if self.last_move is None else (self.last_move[0], self.last_move[1])
        new.captures = {self.BLACK: self.captures[self.BLACK], self.WHITE: self.captures[self.WHITE]}
        return new
