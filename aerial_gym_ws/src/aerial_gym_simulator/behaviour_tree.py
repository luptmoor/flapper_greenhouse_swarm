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
                self.root = SelectorNode(name="RootSelector", depth=0)
            else:
                self.root = SequenceNode(name="RootSequence", depth=0)

            self.root.grow()

        
        #print(f'BT with {self.root.n_children} children created.')

    def set_blackboard(self, blackboard):
        self.blackboard = blackboard

    def feed_forward(self, blackboard):
        feedback, success = self.root.execute(blackboard=blackboard)
        action = torch.zeros(1, 4)
        action[0, 0] = feedback["vx"]
        action[0, 1] = feedback["vy"]
        action[0, 2] = feedback["vz"]
        action[0, 3] = feedback["psi"]
        print(action.shape)

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
    def __init__(self, name, action, value):
        super().__init__(name)
        self.action = action
        self.value = value

    def to_dict(self):
        return {"type": self.__class__.__name__, "name": self.name, "action": self.action, "value": self.value}
    
    def execute(self, blackboard):
        print(f"{self.name}: Set {self.action} to {self.value}.")
        return {self.action: self.value}, True

class ConditionNode(BTNode):
    """Represents a condition check in the behavior tree."""
    def __init__(self, name, reading, operator, value):
        super().__init__(name)
        self.reading = reading
        self.operator = operator
        self.value = value

    def to_dict(self):
        return {"type": self.__class__.__name__, "name": self.name, "reading": self.reading, "operator": self.operator,  "value": self.value}

    def execute(self, blackboard):

        if self.reading == 'fruit_visible':
            print(f"{self.name}: checking if {self.reading}")
            return {}, blackboard[self.reading]
        else:
            print(f"{self.name}: checking if {self.reading} {self.operator} {self.value}.")
            if self.operator == 'greaterThan':
                if blackboard[self.reading] > self.value: return {}, True
                else: return {}, False

            elif self.operator == 'smallerThan':
                if blackboard[self.reading] < self.value: return {}, True
                else: return {}, False

            else:
                print(f'[ERROR] Invalid opeerator "{self.operator}" in {self.name}!')
                return {}, False

class CompositeNode(BTNode):
    """Base class for sequence and selector nodes."""
    def __init__(self, name, depth):
        super().__init__(name)
        self.children = []
        self.depth = depth
        self.feedback = {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "psi": 0.0,
            "message": 0.0,
            "memory": 0.0
        }

    def add_child(self, child):
        """Add a child node, ensuring max children limit is not exceeded."""
        if len(self.children) < 6:  # Limit to 6 children
            self.children.append(child)
        else:
            raise ValueError("Root_sequence0_selector0_action0Max number of children (6) exceeded.")
        
    
    def grow(self):
        for i in range(BT_MAX_CHILDREN):
            die = random.uniform(0, 1)

            # Composite Node
            if die < P_BT_COMPOSITE and self.depth < BT_MAX_DEPTH - 1:
                if random.uniform(0, 1) >= P_BT_SEQUENCE:
                    self.add_child(SelectorNode(self.name + "_selector" + str(i), depth=self.depth+1))
                else:
                    self.add_child(SequenceNode(self.name + "_sequence" + str(i), depth=self.depth+1))
                if self.children[-1].depth < BT_MAX_DEPTH: self.children[-1].grow()    
                
            # Condition Node        
            elif die - P_BT_COMPOSITE < P_BT_CONDITION:
                reading = READING_VARS[random.randint(0, 4)]

                if reading != 'fruit_visible':
                    if random.randint(0, 1) == 1: operator = 'greaterThan'
                    else: operator = 'smallerThan'
                    value = random.uniform(0, 1) * (READING_LIMITS[reading][1] - READING_LIMITS[reading][0]) + READING_LIMITS[reading][0]
                else:
                    operator = 'n.a.'
                    value = 0

                self.add_child(ConditionNode(self.name + "_condition" + str(i), reading=reading, operator=operator, value=value))


            # Action Node
            elif die - P_BT_COMPOSITE - P_BT_CONDITION < P_BT_ACTION:
                action = ACTION_VARS[random.randint(0, 5)]

                if action == 6:
                    print('Neural command based on position')

                value = random.uniform(0, 1) * (ACTION_LIMITS[action][1] - ACTION_LIMITS[action][0]) + ACTION_LIMITS[action][0]
                self.add_child(ActionNode(self.name + "_action" + str(i), action=action, value=value))


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
    
    def execute(self, blackboard):
        print(f'Executing {self.name}.')
        for child in self.children:
            feedback, success = child.execute(blackboard)
            self.feedback.update(feedback)
            if success == False:
                print(f"Feedback of {self.name}: {self.feedback}")
                return self.feedback, False
            
        print(f"Feedback of {self.name}: {self.feedback}")
        return self.feedback, True
        
        

class SelectorNode(CompositeNode):
    """Selector node executes children in order until one succeeds."""

    def execute(self, blackboard):
        print(f"Executing {self.name}.")
        for child in self.children:
            feedback, success = child.execute(blackboard)
            self.feedback.update(feedback)
            if success == True:
                print(f"Feedback of {self.name}: {self.feedback}")
                return self.feedback, True
            
        print(f"Feedback of {self.name}: {self.feedback}")
        return self.feedback, False
