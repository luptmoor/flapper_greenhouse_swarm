import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *
import random
random.seed(BT_SEED)
import json



class BehaviourTree:
    def __init__(self, random_tree=True):

        self.tof_net = ToFNet()
        self.swarm_net = SwarmNet()

        if random_tree:

            # Root node
            if random.uniform(0, 1) >= P_BT_SEQUENCE:
                self.root = SelectorNode(name="RootSelector", depth=0)
            else:
                self.root = SequenceNode(name="RootSequence", depth=0)

            self.root.grow()

        
        #print(f'BT with {self.root.n_children} children created.')

    def feed_forward(self, blackboard):
        feedback, success = self.root.execute(blackboard=blackboard)
        action = torch.zeros(1, 4)
        action[0, 0] = feedback["vx"]
        action[0, 1] = feedback["vy"]
        action[0, 2] = feedback["vz"]
        action[0, 3] = feedback["r"]

        # Swarm net overrides independent velocity control
        if feedback['swarmnet']:
            action = self.swarm_net.forward(blackboard['swarminput'])
            print(f"Aciton determined by SwarmNet: {action}")
        
        # Tof net overrides swarm net (collision avoidance is given a higher priority)
        if feedback['tofnet']:
            action = self.tof_net.forward(blackboard['tofinput'])
            print(f"Aciton determined by ToFNet: {action}")



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
        
        self.action = ACTION_VARS[random.randint(0, 7)]

        if self.action in ['tofnet', 'swarmnet']:
            self.value = True
        else:
            self.value = random.uniform(0, 1) * (ACTION_LIMITS[self.action][1] - ACTION_LIMITS[self.action][0]) + ACTION_LIMITS[self.action][0]


    def to_dict(self):
        return {"type": self.__class__.__name__, "name": self.name, "action": self.action, "value": self.value}
    
    def execute(self, blackboard):
        print(f"{self.name}: Set {self.action} to {self.value}.")
        return {self.action: self.value}, True

class ConditionNode(BTNode):
    """Represents a condition check in the behavior tree."""
    def __init__(self, name):
        super().__init__(name)

        self.reading = READING_VARS[random.randint(0, 2)]

        if self.reading != 'fruit_visible':
            if random.randint(0, 1) == 1: self.operator = 'greaterThan'
            else: self.operator = 'smallerThan'
            self.value = random.uniform(0, 1) * (READING_LIMITS[self.reading][1] - READING_LIMITS[self.reading][0]) + READING_LIMITS[self.reading][0]
        else:
            self.operator = 'n.a.'
            self.value = 0


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
            "r": 0.0,
            "message": 0.0,
            "memory": 0.0,
            "tofnet": False,
            "swarmnet": False
        }

    def add_child(self, child):
        """Add a child node, ensuring max children limit is not exceeded."""
        if len(self.children) < 6:  # Limit to 6 children
            self.children.append(child)
        else:
            raise ValueError("Root_sequence0_selector0_action0Max number of children (6) exceeded.")
        

    def micromutate(self):
        for i in range(len(self.children)):
            if isinstance(self.children[i], ActionNode):
                if random.uniform(0, 1) < P_MICROMUTATION:
                    print(f"Mutation at {self.children[i].name}.")
                    self.children[i] = ActionNode(self.name + "_action" + str(i))

            elif isinstance(self.children[i], ConditionNode):
                if random.uniform(0, 1) < P_MICROMUTATION:
                    print(f"Mutation at {self.children[i].name}.")
                    self.children[i] = ConditionNode(self.name + "_condition" + str(i))

            else:
                self.children[i].micromutate()


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
                self.add_child(ConditionNode(self.name + "_condition" + str(i)))

            # Action Node
            elif die - P_BT_COMPOSITE - P_BT_CONDITION < P_BT_ACTION:
                self.add_child(ActionNode(self.name + "_action" + str(i)))


        #self.n_children = sum([child.n_children for child in self.children if isinstance(child, CompositeNode)])
        


    def to_dict(self):
        """Convert the composite node to a dictionary for saving."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children],
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



## Neural Classes

class ToFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.fc3 = nn.Linear(16, 4)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # sigmoid to keep outputs bounded in (0, 1)

        # Map to correct ranges
        min_tensor = torch.tensor([V_BACKWARD_MAX, V_LEFT_MAX, V_DOWN_MAX, -YAWRATE_MAX])
        max_tensor = torch.tensor([V_FORWARD_MAX, V_RIGHT_MAX, V_UP_MAX, YAWRATE_MAX])

        x = min_tensor + (max_tensor - min_tensor) * x

        return x  # vx, vy, vz, r
    


class SwarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.fc3 = nn.Linear(16, 4)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))   # sigmoid to keep outputs bounded in (0, 1)

        # Map to correct ranges
        min_tensor = torch.tensor([-V_BACKWARD_MAX, -V_LEFT_MAX, -V_DOWN_MAX, -YAWRATE_MAX])
        max_tensor = torch.tensor([V_FORWARD_MAX, V_RIGHT_MAX, V_UP_MAX, YAWRATE_MAX])

        x = min_tensor + (max_tensor - min_tensor) * x


        return x  # vx, vy, vz, r
