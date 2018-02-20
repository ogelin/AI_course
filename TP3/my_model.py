class myAgent(object):
    def __init__(self, gamma=0.99, batch_size=128):
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)

    def act(self, x, epsilon=0.1):
        # TODO
        # fonction utiles: torch.max()
        pass

    def backward(self, transitions):
        batch = Transition(*zip(*transitions))
        # TODO
        # fonctions utiles: torch.gather(), torch.detach()
        # torch.nn.functional.smooth_l1_loss()
        pass
