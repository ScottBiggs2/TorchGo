import torch
from typing import Optional, Tuple, List

class GoGame:
    def __init__(self, BOARD_SIZE):

        self.BLACK = -1
        self.WHITE = 1
        self.EMPTY = 0
        self.BOARD_SIZE = BOARD_SIZE
        self.komi = 6.5
        self.current_player = self.BLACK
        
        self.board = torch.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=torch.int8)
        self.history: List[torch.Tensor] = []
        self.move_log: List[Tuple[int, Optional[int], Optional[int]]] = []
        self.pass_count = 0
        self.game_over = False

        self.last_move: Optional[Tuple[int,int]] = None
        self.captures = {self.BLACK: 0, self.WHITE: 0}
        
        self.history = []
        self.move_log = []
        self.last_move = None
        self.pass_count = 0
        
        # Add move cycle detection
        self.move_cycle = []  # Store last few moves
        self.cycle_threshold = 5  # Number of repeats before ending game
        self.cycle_length = 2  # Length of cycle to detect (e.g., 2 for alternating moves)

    def copy_board(self) -> torch.Tensor:
        return self.board.clone()

    def opponent(self, player: int) -> int:
        return self.BLACK if player == self.WHITE else self.WHITE

    def is_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if self.is_on_board(x + dx, y + dy)]

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
          - move violates Ko (i.e. makes board identical to same player's last move)
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

        # 3) Ko check: check if this move would make the board identical to same player's last move
        if len(self.history) >= 2:
            # Find the last move by the same player
            same_player_last_move = None
            for i in range(len(self.history) - 1, -1, -1):
                if i % 2 == (len(self.history) - 1) % 2:  # Same player's turn
                    same_player_last_move = i
                    break
            
            if same_player_last_move is not None:
                # Simulate the current move
                backup = self.copy_board()
                backup[x, y] = self.current_player
                
                # Remove any captures
                visited = set()
                for nx, ny in self.get_neighbors(x, y):
                    if backup[nx, ny].item() == self.opponent(self.current_player) and (nx, ny) not in visited:
                        group = self._flood_fill_group(nx, ny, backup, visited)
                        if self._count_liberties(group, backup) == 0:
                            for gx, gy in group:
                                backup[gx, gy] = self.EMPTY
                
                # Check if the resulting board matches the same player's last move
                if torch.equal(backup, self.history[same_player_last_move]):
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
        self.remove_dead_stones(self.current_player)

        # Append to history for Ko
        self.history.append(self.copy_board())

        # Log the move
        self.move_log.append((self.current_player, x, y))
        self.pass_count = 0
        
        # Check for move cycles
        self.move_cycle.append((x, y))
        if len(self.move_cycle) >= self.cycle_length * self.cycle_threshold:
            # Check if we have a repeating pattern
            is_cycle = True
            for i in range(self.cycle_length):
                for j in range(self.cycle_threshold):
                    if self.move_cycle[-self.cycle_length * (j+1) + i] != self.move_cycle[-self.cycle_length + i]:
                        is_cycle = False
                        break
                if not is_cycle:
                    break
            
            if is_cycle:
                # print(f"DEBUG: Detected move cycle after {len(self.move_cycle)} moves. Ending game.")
                self.game_over = True
            
            # Keep only the last cycle_length * cycle_threshold moves
            self.move_cycle = self.move_cycle[-self.cycle_length * self.cycle_threshold:]
        
        self.current_player = self.opponent(self.current_player)
        return True

    def estimate_territory(self) -> dict:
        visited = set()
        territory = {self.BLACK: 0, self.WHITE: 0}
        
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i, j].item() == self.EMPTY and (i, j) not in visited:
                    region = set()
                    stack = [(i, j)]
                    borders = set()
                    has_black = False
                    has_white = False
                    
                    while stack:
                        x, y = stack.pop()
                        if (x, y) in visited:
                            continue
                        visited.add((x, y))
                        region.add((x, y))
                        
                        # Get neighbors with bounds checking
                        for nx, ny in self.get_neighbors(x, y):
                            # Skip off-board points
                            if not (0 <= nx < self.BOARD_SIZE and 0 <= ny < self.BOARD_SIZE):
                                continue
                                
                            val = self.board[nx, ny].item()
                            if val == self.EMPTY:
                                if (nx, ny) not in visited:
                                    stack.append((nx, ny))
                            else:
                                if val == self.BLACK:
                                    has_black = True
                                elif val == self.WHITE:
                                    has_white = True
                    
                    # Determine territory ownership
                    if has_black and not has_white:
                        territory[self.BLACK] += len(region)
                    elif has_white and not has_black:
                        territory[self.WHITE] += len(region)
        
        return {'black_territory': territory[self.BLACK],
                'white_territory': territory[self.WHITE]}

    def run_tests(self):
        print("\n===== Running Advanced Go Scoring Tests =====")
        passed = 0
        total = 0
    
        # Test 1: Empty board
        total += 1
        game = GoGame(9)
        terr = game.estimate_territory()
        print("\nTest 1: Empty board")
        game.print_board()
        if terr['black_territory'] == 0 and terr['white_territory'] == 0:
            print("✅ Test 1: Empty board passed")
            passed += 1
        else:
            print("❌ Test 1: Empty board failed")
    
        # Test 2: Fully enclosed territory
        total += 1
        game = GoGame(5)
        # Create black border around board
        for i in range(5):
            game.board[0, i] = game.BLACK
            game.board[4, i] = game.BLACK
            if 0 < i < 4:
                game.board[i, 0] = game.BLACK
                game.board[i, 4] = game.BLACK
        print("\nTest 2: Fully enclosed territory")
        game.print_board()
        terr = game.estimate_territory()
        if terr['black_territory'] == 9 and terr['white_territory'] == 0:
            print("✅ Test 2: Fully enclosed territory passed")
            passed += 1
        else:
            print(f"❌ Test 2: Fully enclosed failed (got {terr})")
    
        # Test 3: Corner enclosure
        total += 1
        game = GoGame(5)
        # Black corner enclosure
        # game.board[0, 0] = game.BLACK
        game.board[0, 1] = game.BLACK
        game.board[1, 0] = game.BLACK
        game.board[1, 1] = game.BLACK
        game.board[4, 1] = game.WHITE
        print("\nTest 3: Corner enclosure")
        game.print_board()
        terr = game.estimate_territory()
        if terr['black_territory'] == 1 and terr['white_territory'] == 0:
            print("✅ Test 3: Corner enclosure passed")
            passed += 1
        else:
            print(f"❌ Test 3: Corner enclosure failed (got {terr})")
    
        # Test 4: Diagonal boundary (should be neutral)
        total += 1
        game = GoGame(5)
        # Create diagonal boundary
        game.board[0, 0] = game.BLACK
        game.board[0, 2] = game.BLACK
        game.board[0, 4] = game.BLACK
        game.board[2, 0] = game.BLACK
        game.board[2, 2] = game.BLACK
        game.board[2, 4] = game.BLACK
        game.board[4, 0] = game.BLACK
        game.board[4, 2] = game.BLACK
        game.board[4, 4] = game.BLACK
        
        game.board[1, 1] = game.WHITE
        game.board[1, 3] = game.WHITE
        game.board[3, 1] = game.WHITE
        game.board[3, 3] = game.WHITE
        print("\nTest 4: Diagonal boundary")
        game.print_board()
        terr = game.estimate_territory()
        if terr['black_territory'] == 0 and terr['white_territory'] == 0:
            print("✅ Test 4: Diagonal boundary passed")
            passed += 1
        else:
            print(f"❌ Test 4: Diagonal boundary failed (got {terr})")
    
        # Test 5: Complex capture sequence
        total += 1
        game = GoGame(5)
        # Create capture ladder
        moves = [
            (0, 0), (1, 0),
            (0, 1), (1, 1),
            (3, 3), (0, 2)
        ]
        
        # . . W .
        # W W . .
        # . . . .
        # . . . B
        # . . . .
        
        for i, (x, y) in enumerate(moves):
            player = game.current_player
            success = game.play_move(x, y)
            if not success:
                print(f"Move {i+1} at ({x},{y}) by {'B' if player == game.BLACK else 'W'} failed!")
                
        print("\nTest 5: Complex capture sequence")
        game.print_board()
        
        print(f"Captures - Black: {game.captures[game.BLACK]}, White: {game.captures[game.WHITE]}")
        terr = game.estimate_territory()
        score = game.score()
        
        # Expect white to capture at least 1 stone
        if game.captures[game.WHITE] >= 1:
            print("✅ Test 5: Capture sequence passed")
            print(f"Scores - Black {score['black_score']}, White {score['white_score']}")
            passed += 1
        else:
            print(f"❌ Test 5: Capture sequence failed (white captures: {game.captures[game.WHITE]})")
            print(f"Scores - Black {score['black_score']}, White {score['white_score']}")

    
        # Test 6: Seki (mutual life) position
        total += 1
        """
        Create seki position:
          B B B . .
          B W B . .
          B B B . .
          . . . . .
          . . . . .
        But with eyes:
          B B B . .
          B . W . .
          B W B . .
          . . . . .
          . . . . .
        Actually create:
          Positions where both groups are alive without two eyes
        """
        # Better seki position
        game = GoGame(5)
        # Create:
        #   . B W .
        #   B . W B
        #   W W . W
        #   . B W .
        game.board[0, 1] = game.BLACK
        game.board[0, 2] = game.WHITE
        game.board[1, 0] = game.BLACK
        game.board[1, 2] = game.WHITE
        game.board[1, 3] = game.BLACK
        game.board[2, 0] = game.WHITE
        game.board[2, 1] = game.WHITE
        game.board[2, 3] = game.WHITE
        game.board[3, 1] = game.BLACK
        game.board[3, 2] = game.WHITE

        print("\nTest 6: Seki position")
        game.print_board()
        terr = game.estimate_territory()
        # Center points should be neutral
        if terr['black_territory'] == 1 and terr['white_territory'] == 1:
            print("✅ Test 6: Seki position passed")
            passed += 1
        else:
            print(f"❌ Test 6: Seki position failed (got {terr})")
    
        # Test 7: Double enclosure
        total += 1
        game = GoGame(5)
        # Create two separate enclosures
        # Black top-left, white bottom-right
        for i in range(3):
            for j in range(3):
                if i == 0 or j == 0:
                    game.board[i, j] = game.BLACK
                if i == 4 or j == 4:
                    game.board[i, j] = game.WHITE
        game.board[0, 4] = game.EMPTY  # Don't connect
        game.board[4, 0] = game.EMPTY  # Don't connect
        
        # Add stones to define territories
        game.board[0, 3] = game.BLACK
        game.board[1, 4] = game.WHITE
        game.board[3, 0] = game.BLACK
        game.board[4, 1] = game.WHITE
        
        print("\nTest 7: Double enclosure")
        game.print_board()
        terr = game.estimate_territory()
        # Expect black territory in top-left, white in bottom-right
        if terr['black_territory'] == 0 and terr['white_territory'] == 0:
            print("✅ Test 7: Double enclosure passed")
            passed += 1
        else:
            print(f"❌ Test 7: Double enclosure failed (got {terr})")
    
        # Test 8: Captured stone in territory
        total += 1
        game = GoGame(5)
        # Black surrounds white stone
        moves = [
            (1, 1), (0, 0),  # B, W
            (0, 1), (1, 0),  # B, W
            (0, 2), (2, 0),  # B, W
            (1, 2), (0, 3),  # B, W
            (2, 1), (3, 0),  # B, W
            (2, 2), (1, 3),  # B, W
            (3, 1), (2, 3),  # B, W
            (3, 2),          # B (captures white)
        ]
        for x, y in moves:
            game.play_move(x, y)
        print("\nTest 8: Captured stone in territory")
        game.print_board()
        terr = game.estimate_territory()
        score = game.score()
        # Captured stone should count as territory
        if terr['black_territory'] >= 1 and game.captures[game.BLACK] >= 1:
            print("✅ Test 8: Captured stone passed")
            passed += 1
        else:
            print(f"❌ Test 8: Captured stone failed (terr: {terr}, captures: {game.captures})")

        
        # test 9, a diagonal boundary not in a corner
        total += 1
        game = GoGame(5)
        
        game.board[0, 0] = game.BLACK
        game.board[0, 1] = game.WHITE
        
        game.board[1, 1] = game.BLACK
        game.board[1, 2] = game.WHITE
        
        game.board[2, 2] = game.BLACK
        game.board[2, 3] = game.WHITE
        
        game.board[3, 3] = game.BLACK
        game.board[3, 4] = game.WHITE
        
        game.board[4, 4] = game.BLACK

        
        print("\nTest 9: Diagonal territory")
        game.print_board()
        terr = game.estimate_territory()
        score = game.score()
        # Captured stone should count as territory
        if terr['black_territory'] == 10 and terr['white_territory'] == 6:
            print("✅ Test 9: Diagonal wall passed")
            passed += 1
        else:
            print(f"❌ Test 9: Diagonal wall failed (terr: {terr}, captures: {game.captures})")
    
        print(f"\nTests passed: {passed}/{total}")
        if passed == total:
            print("🎉 All tests passed successfully!")
        else:
            print("⚠️ Some tests failed, check implementation")
    

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
