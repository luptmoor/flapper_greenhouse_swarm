import torch
from settings import *
import random
random.seed(BT_SEED)
import json



class BehaviourTree:
    def __init__(self, random_tree=True):

        if random_tree:

            # Root node
            if random.uniform(0, 1) >= P_BT_SEQUENCE:
                self.root = SelectorNode(name="Root", depth=0)
            else:
                self.root = SequenceNode(name="Root", depth=0)

            self.root.grow()

        
        print(f'BT with {self.root.n_children} children created.')

    def feed_forward(self, blackboard):
        action = torch.zeros(2);
        return action
    
    def save2file(self, filename="behavior_tree.json"):
        """Save the behavior tree to a JSON file."""
        with open(filename, "w") as file:
            json.dump(self.root.to_dict(), file, indent=4)



    ### Node classes


class BTNode:
    """Base class for all behavior tree nodes."""
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        """Convert the node to a dictionary for saving."""
        return {"type": self.__class__.__name__, "name": self.name}

class ActionNode(BTNode):
    """Represents an action in the behavior tree."""
    def __init__(self, name):
        super().__init__(name)

class ConditionNode(BTNode):
    """Represents a condition check in the behavior tree."""
    def __init__(self, name):
        super().__init__(name)

class CompositeNode(BTNode):
    """Base class for sequence and selector nodes."""
    def __init__(self, name, depth):
        super().__init__(name)
        self.children = []
        self.depth = depth

    def add_child(self, child):
        """Add a child node, ensuring max children limit is not exceeded."""
        if len(self.children) < 6:  # Limit to 6 children
            self.children.append(child)
        else:
            raise ValueError("Max number of children (6) exceeded.")
    
    def grow(self):
        for i in range(BT_MAX_CHILDREN):
            die = random.uniform(0, 1)
            if die < P_BT_COMPOSITE and self.depth < BT_MAX_DEPTH - 1:
                if random.uniform(0, 1) >= P_BT_SEQUENCE:
                    self.add_child(SelectorNode(self.name + "_selector" + str(i), depth=self.depth+1))
                else:
                    self.add_child(SequenceNode(self.name + "_sequence" + str(i), depth=self.depth+1))
                if self.children[-1].depth < BT_MAX_DEPTH: self.children[-1].grow()    
                
                    
            elif die - P_BT_COMPOSITE < P_BT_CONDITION:
                self.add_child(ConditionNode(self.name + "_condition" + str(i)))

            elif die - P_BT_COMPOSITE - P_BT_CONDITION < P_BT_ACTION:
                self.add_child(ActionNode(self.name + "_action" + str(i)))
        self.n_children = sum([child.n_children for child in self.children if isinstance(child, CompositeNode)])
        


    def to_dict(self):
        """Convert the composite node to a dictionary for saving."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children],
            "n_children": self.n_children
        }

class SequenceNode(CompositeNode):
    """Sequence node executes children in order until one fails."""
    pass

class SelectorNode(CompositeNode):
    """Selector node executes children in order until one succeeds."""
    pass
