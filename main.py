import tkinter as tk
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from minimax_ai import MiniMaxAI
from torch_ai import TicTacToeAI, choose_move, train_ai

size_of_board = 600
symbol_size = (size_of_board / 3 - size_of_board / 8) / 2
symbol_thickness = 50
symbol_X_color = '#EE4035'
symbol_O_color = '#0492CF'
Green_color = '#7BC043'


class Tic_Tac_Toe():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Tic-Tac-Toe')
        self.canvas = tk.Canvas(self.window, width=size_of_board, height=size_of_board)
        self.canvas.pack()
        self.window.bind('<Button-1>', self.click)

        self.initialize_board()
        self.player_X_turns = True
        self.board_status = np.zeros(shape=(3, 3))

        self.player_X_starts = True
        self.reset_board = False
        self.gameover = False
        self.tie = False
        self.X_wins = False
        self.O_wins = False

        self.X_score = 0
        self.O_score = 0
        self.tie_score = 0

        # Inicializáljuk az AI-t
        self.ai_model = TicTacToeAI()
        self.optimizer = optim.Adam(self.ai_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.5  # Epsilon-greedy stratégia

        # Tanítási adatok
        self.game_data = []
        self.train_data = []

    def mainloop(self):
        self.window.mainloop()

    def initialize_board(self):
        self.canvas.delete("all")
        for i in range(2):
            self.canvas.create_line((i + 1) * size_of_board / 3, 0, (i + 1) * size_of_board / 3, size_of_board)
        for i in range(2):
            self.canvas.create_line(0, (i + 1) * size_of_board / 3, size_of_board, (i + 1) * size_of_board / 3)

    def play_again(self):
        self.initialize_board()
        self.player_X_starts = True
        self.player_X_turns = self.player_X_starts
        self.board_status = np.zeros(shape=(3, 3))
        self.reset_board = False
        self.gameover = False
        self.tie = False
        self.X_wins = False
        self.O_wins = False
        self.game_data = []  # Töröljük a korábbi játék adatait

    def draw_O(self, logical_position):
        logical_position = np.array(logical_position)
        grid_position = self.convert_logical_to_grid_position(logical_position)
        self.canvas.create_oval(grid_position[0] - symbol_size, grid_position[1] - symbol_size,
                                grid_position[0] + symbol_size, grid_position[1] + symbol_size, width=symbol_thickness,
                                outline=symbol_O_color)

    def draw_X(self, logical_position):
        grid_position = self.convert_logical_to_grid_position(logical_position)
        self.canvas.create_line(grid_position[0] - symbol_size, grid_position[1] - symbol_size,
                                grid_position[0] + symbol_size, grid_position[1] + symbol_size, width=symbol_thickness,
                                fill=symbol_X_color)
        self.canvas.create_line(grid_position[0] - symbol_size, grid_position[1] + symbol_size,
                                grid_position[0] + symbol_size, grid_position[1] - symbol_size, width=symbol_thickness,
                                fill=symbol_X_color)

    def display_gameover(self):
        if self.X_wins:
            self.X_score += 1
            text = 'Winner: Player 1 (X)'
            color = symbol_X_color
            reward = -1  # AI veszített
        elif self.O_wins:
            self.O_score += 1
            text = 'Winner: AI (O)'
            color = symbol_O_color
            reward = 1  # AI nyert
        else:
            self.tie_score += 1
            text = 'Its a tie'
            color = 'gray'
            reward = 0  # Döntetlen

        # Frissítsük a tanítási adatokat a jutalommal
        for state, action in self.game_data:
            self.train_data.append((state, action, reward))

        self.canvas.delete("all")
        self.canvas.create_text(size_of_board / 2, size_of_board / 3, font="cmr 40 bold", fill=color, text=text)

        score_text = 'Scores \n'
        self.canvas.create_text(size_of_board / 2, 5 * size_of_board / 8, font="cmr 40 bold", fill=Green_color,
                                text=score_text)

        score_text = f'Player 1 (X): {self.X_score}\n'
        score_text += f'AI (O): {self.O_score}\n'
        score_text += f'Tie: {self.tie_score}'
        self.canvas.create_text(size_of_board / 2, 3 * size_of_board / 4, font="cmr 30 bold", fill=Green_color,
                                text=score_text)

        score_text = 'Click to play again'
        self.canvas.create_text(size_of_board / 2, 15 * size_of_board / 16, font="cmr 20 bold", fill="gray",
                                text=score_text)

        self.reset_board = True

        # Tanítsuk a modellt a játék végén
        self.train_ai_model()

    def train_ai_model(self):
        if not self.train_data:
            return

        states, actions, rewards = zip(*self.train_data)
        for state, action, reward in zip(states, actions, rewards):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target = torch.zeros(9)
            if action is not None:
                target[action] = reward
            target_tensor = target.unsqueeze(0)

            self.optimizer.zero_grad()
            output = self.ai_model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            # Kiírjuk a veszteséget
            print(f"Loss: {loss.item()}")

        self.train_data = []  # Töröljük a tanítási adatokat a következő játékhoz

    def convert_logical_to_grid_position(self, logical_position):
        logical_position = np.array(logical_position, dtype=int)
        return (size_of_board / 3) * logical_position + size_of_board / 6

    def convert_grid_to_logical_position(self, grid_position):
        grid_position = np.array(grid_position)
        return np.array(grid_position // (size_of_board / 3), dtype=int)

    def is_grid_occupied(self, logical_position):
        return self.board_status[logical_position[0]][logical_position[1]] != 0

    def is_winner(self, player_mark):
        player = -1 if player_mark == 'X' else 1

        # Three in a row
        for i in range(3):
            if self.board_status[i][0] == self.board_status[i][1] == self.board_status[i][2] == player:
                return True
            if self.board_status[0][i] == self.board_status[1][i] == self.board_status[2][i] == player:
                return True

        # Diagonals
        if self.board_status[0][0] == self.board_status[1][1] == self.board_status[2][2] == player:
            return True
        if self.board_status[0][2] == self.board_status[1][1] == self.board_status[2][0] == player:
            return True

        return False

    def is_tie(self):
        return np.all(self.board_status != 0)

    def is_gameover(self):
        if self.X_wins or self.O_wins or self.is_tie():
            return True
        return False

    def click(self, event):
        grid_position = [event.x, event.y]
        logical_position = self.convert_grid_to_logical_position(grid_position)

        if not self.reset_board and not self.gameover:
            if self.player_X_turns:
                if not self.is_grid_occupied(logical_position):
                    self.draw_X(logical_position)
                    self.board_status[logical_position[0]][logical_position[1]] = -1
                    self.player_X_turns = not self.player_X_turns
                    self.window.update_idletasks()

                    # Ellenőrizzük, hogy a játékos nyert-e
                    if self.is_winner('X'):
                        self.X_wins = True
                        self.gameover = True
                        self.display_gameover()
                        return
            
            # AI lépése
            if not self.player_X_turns and not self.reset_board and not self.gameover:
                flat_board = [0 if cell == 0 else (1 if cell == 1 else -1) for cell in self.board_status.flatten()]
                move = choose_move(self.ai_model, flat_board, self.epsilon)
                if move is not None:
                    logical_position = divmod(move, 3)
                    self.draw_O(logical_position)
                    self.board_status[logical_position[0]][logical_position[1]] = 1
                    self.player_X_turns = not self.player_X_turns
                    self.window.update_idletasks()

                    # Jegyezzük fel a lépést a tanításhoz
                    self.game_data.append((flat_board, move))

                    # Tanítsuk az AI-t az aktuális lépés alapján
                    self.train_ai_model()

                    # Ellenőrizzük, hogy az AI nyert-e
                    if self.is_winner('O'):
                        self.O_wins = True
                        self.gameover = True
                        self.display_gameover()
                        return

                # Ellenőrizzük, hogy döntetlen lett-e
                if self.is_tie():
                    self.tie = True
                    self.gameover = True
                    self.display_gameover()
                    return

        else:
            self.play_again()

game_instance = Tic_Tac_Toe()
game_instance.mainloop()