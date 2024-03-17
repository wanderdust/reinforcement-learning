import lightning as L
from q_function import q_function
from torch import optim


class DQN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.q_function = q_function()

    def training_step(self, batch, batch_idx):
        # 1. Take action based on policy and observe r + x_t+1
        # if prob <= 0.1 random action, else use Q-function

        # 2. Preprocess x_t+1

        # 3. Store preprocessed state in memory (x, a, r, x_t+1)

        # 4. Sample random batch of transitions

        # 5. Set target `y = r + discount * max_a Q(next_state)` or `y=r` if final state

        # Calculate loss `(y - Q(state, action))**2)`
        # return loss

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=0.01)
        return optimizer
