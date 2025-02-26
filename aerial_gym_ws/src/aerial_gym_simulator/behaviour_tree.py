import torch

class BehaviourTree:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes;

    def feed_forward(self, blackboard):
        action = torch.zeros(2);
        return action
    