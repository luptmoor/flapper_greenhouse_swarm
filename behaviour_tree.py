import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *
import random
import json
from graphviz import Digraph
from action_modules import *

shapedict = {
    'ActionNode': 'box',
    'ConditionNode': 'ellipse',
    'SelectorNode': 'circle',
    'SequenceNode': 'box'
}

labeldict = {
    'ActionNode': 'Action',
    'ConditionNode': 'Condition',
    'SelectorNode': '?',
    'SequenceNode': '-->'
}

operatordict = {
    'greaterThan': '>',
    'smallerThan': '<',
    'n.a.': '?'
}

colordict = {
    'running': 'lightblue',
    'success': 'green',
    'failure': 'red',
    'idle': 'white'
}

action_strings = [
    'Approach',
    'Avoid other drones',
    'Turn left',
    'Follow wall',
    'Random Walk',
    'Disperse',
]

actions = [
    approach,
    apf_avoidance,
    clear_path,
    follow_wall,
    random_walk,
    disperse
]

condition_strings = [
    'Fruit visible?',
    '# discovered fruit > X ?',
    '# new fruit last 30s < X ?',
    'Path clear?',
    'Minimum peer distance < X ?'
]

conditions = [
    fruit_visible,
    fruit_counter,
    discovery_rate,
    path_clear,
    min_distance
]



def dict_to_bt(data):
    """Recursively reconstruct a behavior tree from a dictionary."""
    node_type = data["type"]
    
    # Reconstruct an ActionNode: restore name, action and value.
    if node_type == "ActionNode":
        node = ActionNode(data["name"])
        if "action" in data:
            node.action = data["action"]
        if "value" in data:
            node.value = data["value"]
        return node
    
    # Reconstruct a ConditionNode: restore name, reading, operator and value.
    elif node_type == "ConditionNode":
        node = ConditionNode(data["name"])
        if "reading" in data:
            node.reading = data["reading"]
        if "operator" in data:
            node.operator = data["operator"]
        if "value" in data:
            node.value = data["value"]
        return node
    
    # Reconstruct composite nodes (SequenceNode or SelectorNode)
    elif node_type in ["SequenceNode", "SelectorNode"]:
        # Try to read depth from the data; default to 0 if not provided.
        depth = data.get("depth", 0)
        if node_type == "SequenceNode":
            node = SequenceNode(data["name"], depth)
        else:
            node = SelectorNode(data["name"], depth)
        
        # Process children recursively.
        for child_data in data.get("children", []):
            child_node = dict_to_bt(child_data)
            node.add_child(child_node)
        return node

    else:
        raise ValueError(f"Unknown node type: {node_type}")





class BehaviourTree:
    def __init__(self, seed=SEED):
        random.seed(seed)
        
        self.fitness = 0
        self.root = None

       
        if random.uniform(0, 1) >= P_BT_SEQUENCE:
            self.root = SelectorNode(name="RootSelector", depth=0)
        else:
            self.root = SequenceNode(name="RootSequence", depth=0)

        self.root.grow()
        
    

    def load_from_file(self, path):
        """Load a behavior tree from a JSON file."""
        with open(path, "r") as file:
            tree_dict = json.load(file)
        self.root = dict_to_bt(tree_dict)


    def save_to_json(self, path):
        """Save the behavior tree to a JSON file."""
        with open(path, "w") as file:
            json.dump(self.root.to_dict(), file, indent=4)
    

    def feed_forward(self, blackboard):
        feedback, success = self.root.execute(blackboard=blackboard)

        # Swarm net overrides independent velocity control
        # if feedback['swarmnet']:
        #     feedback["vx"], feedback["vz"], feedback["r"] = self.swarm_net.forward(blackboard['swarminput'])
        #     print(f"Action determined by SwarmNet: {feedback['vx']}, {feedback['vz']}, {feedback['r']}")
        
        # # Tof net overrides swarm net (collision avoidance is given a higher priority)
        # if feedback['tofnet']:
        #     feedback["vx"], feedback["vz"], feedback["r"] = self.tof_net.forward(blackboard['tofinput'])
        #     print(f"Action determined by ToFNet: {feedback['vx']}, {feedback['vz']}, {feedback['r']}")

        return feedback["vx"], feedback["vz"], feedback["r"]



    def plot(self):
        """
        Visualize the behavior tree using Graphviz.
        """
        dot = Digraph(comment='Behavior Tree')
        dot.attr(nodesep='0.3', ranksep='1.5')

        def add_nodes_edges(node, parent_id=None):
            # Create a unique id for each node (using the id() built-in is one option)
            node_id = str(id(node))
            # Display node type and name (additional details can be added if desired)
            classname = f"{node.__class__.__name__}"
            shape = shapedict[classname]
            label = labeldict[classname]
            fillcolour = colordict[node.state]

            if hasattr(node, 'action_string'):
                label += f"\n {node.action_string}"
            if hasattr(node, 'condition_string'):
                label += f"\n{node.condition_string}"
                

            # Add current node.
            dot.node(node_id, label, shape=shape, fillcolor=fillcolour, style='filled')

            # If there's a parent, add an edge from parent to current node.
            if parent_id is not None:
                dot.edge(parent_id, node_id)

            # If node is a composite node, traverse its children.
            if hasattr(node, 'children'):
                for child in node.children:
                    add_nodes_edges(child, node_id)

        add_nodes_edges(self.root)

        # Render and view the graph; file formats can be 'pdf', 'png', etc.
        #dot.render(path, view=True, format='pdf')
        return dot
    

    def save_to_pdf(self, path):
        """
        Visualize the behavior tree using Graphviz.
        """
        dot = Digraph(comment='Behavior Tree')
        dot.attr(nodesep='0.3', ranksep='1.5')

        def add_nodes_edges(node, parent_id=None):
            # Create a unique id for each node (using the id() built-in is one option)
            node_id = str(id(node))
            # Display node type and name (additional details can be added if desired)
            classname = f"{node.__class__.__name__}"
            shape = shapedict[classname]
            label = labeldict[classname]

            if hasattr(node, 'action_string'):
                label += f"\n {node.action_string}"
            if hasattr(node, 'condition_string'):
                label += f"\n{node.condition_string}"
                

            # Add current node.
            dot.node(node_id, label, shape=shape)

            # If there's a parent, add an edge from parent to current node.
            if parent_id is not None:
                dot.edge(parent_id, node_id)

            # If node is a composite node, traverse its children.
            if hasattr(node, 'children'):
                for child in node.children:
                    add_nodes_edges(child, node_id)

        add_nodes_edges(self.root)

        # Render and view the graph; file formats can be 'pdf', 'png', etc.
        dot.render(path, view=True, format='pdf')


### Node classes ############################

class BTNode:
    """Base class for all behavior tree nodes."""
    def __init__(self, name):
        self.name = name
        self.state = 'idle'

    def to_dict(self):
        """Convert the node to a dictionary for saving."""
        return {"type": self.__class__.__name__, "name": self.name}



class ActionNode(BTNode):
    """Represents an action in the behavior tree."""
    def __init__(self, name):
        super().__init__(name)
        
        action_id = random.choice(range(len(actions)))
        self.action_string = action_strings[action_id]
        self.action = actions[action_id]


    def to_dict(self):
        return {"type": self.__class__.__name__, "name": self.name, "action": self.action, "value": self.value}
    

    def execute(self, blackboard):
        vx, vz, r, self.state = self.action(blackboard)
        feedback = {
            "vx": vx,
            "vz": vz,
            "r": r,
        }   

        return feedback, self.state



class ConditionNode(BTNode):
    """Represents a condition check in the behavior tree."""
    def __init__(self, name):
        super().__init__(name)

        condition_id = random.choice(range(len(conditions)))
        self.condition_string = condition_strings[condition_id]
        self.condition = conditions[condition_id]


    def to_dict(self):
        return {"type": self.__class__.__name__, "name": self.name, "reading": self.reading, "operator": self.operator,  "value": self.value}

    def execute(self, blackboard):
        self.state = self.condition(blackboard)
        return {}, self.state



class CompositeNode(BTNode):
    """Base class for sequence and selector nodes."""
    def __init__(self, name, depth):
        super().__init__(name)
        self.children = []
        self.depth = depth
        self.feedback = {
            "vx": 0.0,
            "vz": 0.0,
            "r": 0.0,
        }

    def add_child(self, child):
        """Add a child node, ensuring max children limit is not exceeded."""
        if len(self.children) < 6:  # Limit to 6 children
            self.children.append(child)
        else:
            raise ValueError("Root_sequence0_selector0_action0Max number of children (6) exceeded.")
        

    def clear(self):
        self.children = []


    def macromutate(self):
        for i in range(len(self.children)):
            if isinstance(self.children[i], CompositeNode):
                if random.uniform(0, 1) < P_MACROMUTATION:
                    print(f'Macromutation at {self.children[i].name}')
                    self.children[i].clear()
                    self.children[i].grow()

                else:
                    self.children[i].macromutate()


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
                if self.type == 'selector':
                    break


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
    def __init__(self, name, depth):
        super().__init__(name, depth)
        self.type = 'sequence'
    
    def execute(self, blackboard):
        #print(f'Executing {self.name}.')
        for child in self.children:
            feedback, success = child.execute(blackboard)
            self.feedback.update(feedback)
            if success == 'failure':
                self.state = 'failure'
                #print(f"Feedback of {self.name}: {self.feedback}")
                return self.feedback, 'failure'
            if success == 'running':
                self.state = 'running'
                #print(f"Feedback of {self.name}: {self.feedback}")
                return self.feedback, 'running'
            
        #print(f"Feedback of {self.name}: {self.feedback}")
        self.state = 'success'
        return self.feedback, 'success'       



class SelectorNode(CompositeNode):
    """Selector node executes children in order until one succeeds."""
    def __init__(self, name, depth):
        super().__init__(name, depth)
        self.type = 'selector'

    def execute(self, blackboard):
        #print(f"Executing {self.name}.")
        for child in self.children:
            feedback, success = child.execute(blackboard)
            self.feedback.update(feedback)
            if success == 'success':
                self.state = 'success'
                #print(f"Feedback of {self.name}: {self.feedback}")
                return self.feedback, 'success'
            if success == 'running':
                self.state = 'running'
                #print(f"Feedback of {self.name}: {self.feedback}")
                return self.feedback, 'running'
            
        #print(f"Feedback of {self.name}: {self.feedback}")
        self.state = 'failure'
        return self.feedback, 'failure'

