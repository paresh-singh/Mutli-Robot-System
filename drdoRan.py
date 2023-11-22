import numpy as np
import matplotlib.pyplot as plt

# Step 1: Represent the Graph
num_vertices = 10
idleness = np.full(num_vertices,20)
mean_idleness = np.full(num_vertices,20)
graph = np.zeros((num_vertices, num_vertices))
importance = [5, 8, 6, 8, 7, 6, 7, 5, 5, 9] #fixed
agents_passed_through = [[] for _ in range(num_vertices)]
edges = [(0, 5), (0, 1), (1, 7), (2, 5), (2, 3), (3, 8), (3, 9), (4, 6), (4, 7), (4, 9), (8, 9), (7, 8), (5, 6),
         (5, 0), (1, 0), (7, 1), (5, 2), (3, 2), (8, 3), (9, 3), (6, 4), (7, 4), (9, 4), (9, 8), (8, 7), (6, 5)]
mycolor = ["blue","green","yellow","black"]
for edge in edges:
    i, j = edge
    graph[j, i] = 1
    graph[i, j] = 1 

# Step 2: Initialize Agents
num_agents = 4
agent_positions = [0, 9, 3, 7]  # fixed 
temp = 5  # You can adjust the temperature parameter T as needed
for val in agent_positions : 
    idleness[val] -= 5

final_value = np.zeros(num_vertices)

def visualize_graph(agent_positions, ct, ax):
    ax.set_title(f"Graph Visualization of {ct}")
    for edge in edges:
        i, j = edge
        x = [i % 5, j % 5]
        y = [i // 5, j // 5]
        ax.plot(x, y, "b-", linewidth=1)
    for i in range(num_vertices):
        x = i % 5
        y = i // 5
        ax.scatter(x, y, s=100, color="red", label=f"Node {i}")
    for i, pos in enumerate(agent_positions):
        x = pos % 5
        y = pos // 5
        ax.scatter(x, y, s=100, color=mycolor[i], label=f"Agent {i}")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
# Step 3: Define Utility Computation Function
def vertex_idleness(node, time_step):
    mean_idleness[node] = mean_idleness[node] + ( idleness[node] - mean_idleness[node])/time_step
    return mean_idleness[node]
def graph_idleness():
    total_idleness = sum(mean_idleness)
    num_vertices = len(idleness)
    return total_idleness / num_vertices
def travel_cost(node, node_agent):
    if graph[node,node_agent]==1 or abs(node-node_agent)==0: return 1 
    return abs(node-node_agent)     
def compute_utility(node,agent):
    if idleness[node] < 6 or node == agent_positions[agent]: return 0  
    sum_inverse_cost = sum(1 / travel_cost(node, agent_positions[j]) for j in range(num_agents) if j != agent)  
    utility = importance[node] * idleness[node] - travel_cost(node, agent_positions[agent]) / sum_inverse_cost
    return round(utility,2)

def start(ct): 
    print(f"\n This is the {ct} iteration ") 
    global temp
    time_step = ct+1 
    # Step 4: Phase 1 - Utility Computation
    utilities = []
    for agent in range(num_agents):
        current_position = agent_positions[agent]
        for node in range(num_vertices):
            if graph[current_position, node] == 1 and node != current_position:
                utilities.append([node, agent, compute_utility(node,agent)])                      
    sorted_utilities = sorted(utilities, key=lambda x: x[2], reverse=True)
    avg_graph_idleness = graph_idleness()
    print(f"Average Graph Idleness at iteration {ct}: {avg_graph_idleness}")
    for vertex in range(num_vertices):
        avg_vertex_idleness = vertex_idleness(vertex, time_step)
        print(f"Vertex {vertex} avg Idleness  {avg_vertex_idleness} with idleness of {idleness[vertex]}")
    # Step 5: Phase 2 - Assignment with Consensus
    assignments = []
    assigned_actions = set()

    for i in range(num_agents):
        utilities_i, nodes_i = zip(*[(util, node) for node, agent, util in sorted_utilities if agent == i])
        action_probabilities = [np.exp(utility_i / temp) for utility_i in utilities_i]
        action_probabilities_sum = sum(action_probabilities)
        normalized_probabilities = [prob / action_probabilities_sum for prob in action_probabilities]
        prob_nodes = []
        for j in range(len(normalized_probabilities)):
            prob_nodes.append([normalized_probabilities[j], nodes_i[j]])
        sorted_prob_nodes = sorted(prob_nodes, key=lambda x: x[0], reverse=True)
        
        selected_node = np.random.choice([node for prob, node in sorted_prob_nodes], p=normalized_probabilities)
        while selected_node in assigned_actions:
            selected_node = np.random.choice([node for prob, node in sorted_prob_nodes], p=normalized_probabilities)
        agents_passed_through[selected_node].append(i)
        assignments.append([selected_node, i])
        assigned_actions.add(selected_node)
                        
    visited = [False]*num_vertices
    for node, i in assignments:
        final_value[node] += 1 
        visited[node]= True
        idleness[node] -= 5 
        agent_positions[i] = node 
        print(f"Agent {i} assigned actions: Node {node} ")
    
    for x in range(num_vertices): 
        if not visited[x]:
            idleness[x] += 5 
    print("\n\n")
 
for i in range(20):
    fig, ax = plt.subplots()
    start(i)
    visualize_graph(agent_positions, i, ax)
    plt.tight_layout()
    plt.show()

for node, agents in enumerate(agents_passed_through):
    print(f"Node {node}: Agents Passed Through: {agents}")
for i in range(len(final_value)):
    print(f"{i} is visited {final_value[i]} and importance is {importance[i]}")
