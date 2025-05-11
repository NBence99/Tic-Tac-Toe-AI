import math

class MiniMaxAI:
    def __init__(self, player, opponent):
        self.player = player
        self.opponent = opponent

    def minimax(self, board, depth, is_maximizing, alpha=-math.inf, beta=math.inf, max_depth=5):
        if depth == max_depth:
            return 0 

        # Debug: Kiírjuk az aktuális táblaállapotot
        print(f"Depth: {depth}, Is Maximizing: {is_maximizing}")
        #print("Current Board State:")
        #print([board[i:i+3] for i in range(0, len(board), 3)])  # 3x3-as formátumban jelenítjük meg

        winner = self.check_winner(board)
        if winner == self.player:
            return 10 - depth
        elif winner == self.opponent:
            return depth - 10
        elif self.is_board_full(board):
            return 0

        if is_maximizing:
            best_score = -math.inf
            for move in self.get_available_moves(board):
                board[move] = self.player
                score = self.minimax(board, depth + 1, False, alpha, beta, max_depth)
                board[move] = None
                best_score = max(best_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = math.inf
            for move in self.get_available_moves(board):
                board[move] = self.opponent
                score = self.minimax(board, depth + 1, True, alpha, beta, max_depth)
                board[move] = None
                best_score = min(best_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_score

    def find_best_move(self, board):
        best_score = -math.inf
        best_move = None
        for move in self.get_available_moves(board):
            board[move] = self.player
            score = self.minimax(board, 0, False)
            board[move] = None
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def check_winner(self, board):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] is not None:
                return board[combo[0]]
        return None

    def is_board_full(self, board):
        return all(cell is not None for cell in board)

    def get_available_moves(self, board):
        return [i for i, cell in enumerate(board) if cell is None]