import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TicTacToeAI(nn.Module):
    def __init__(self):
        super(TicTacToeAI, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation, raw scores)
        return x

def choose_move(ai, state, epsilon=0.1):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        output = ai(state_tensor)
    
    # Epsilon-greedy stratégia
    if np.random.rand() < epsilon:
        # Véletlenszerű lépés
        empty_indices = [i for i, cell in enumerate(state) if cell == 0]
        if empty_indices:
            move = np.random.choice(empty_indices)
            return move
        else:
            return None  # Nincs több lépés
    else:
        # Legjobb lépés választása
        empty_indices = [i for i, cell in enumerate(state) if cell == 0]
        if not empty_indices:
            return None
        valid_outputs = output[0][empty_indices]
        best_move_index = torch.argmax(valid_outputs).item()
        move = empty_indices[best_move_index]
        return move

def train_ai(ai, optimizer, criterion, game_data, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        for state, action, reward in game_data:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target = torch.zeros(9)
            if action is not None:
                target[action] = reward
            target_tensor = target.unsqueeze(0)

            optimizer.zero_grad()
            output = ai(state_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(game_data)}")

        # Kiírjuk az aktuális veszteséget minden epóka után
        print(f"Epoch {epoch}, Total Loss: {total_loss}")

# Example usage
if __name__ == "__main__":
    ai = TicTacToeAI()
    optimizer = optim.Adam(ai.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Example training data (state, action, reward)
    # State: 1D array of 9 elements (0 = empty, 1 = AI, -1 = opponent)
    # Action: Integer representing the move (0-8)
    # Reward: Float representing the reward for the action
    game_data = [
        ([0, 0, 0, 0, 1, 0, 0, -1, 0], 6, 1.0),
        ([1, -1, 0, 0, 1, 0, 0, -1, 0], 2, 0.5),
    ]

    train_ai(ai, optimizer, criterion, game_data)

    # Example game state
    current_state = [0, 0, 0, 0, 1, 0, 0, -1, 0]
    move = choose_move(ai, current_state)
    print(f"AI chooses move: {move}")