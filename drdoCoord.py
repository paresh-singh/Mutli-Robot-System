import numpy as np
import matplotlib.pyplot as plt

# Step 1: Represent the Graph
num_vertices = 10
idleness = np.full(num_vertices,20)
mean_idleness = np.full(num_vertices,20)
graph = np.zeros((num_vertices, num_vertices))
importance = [4, 9, 6, 4, 9, 3, 8, 5, 5, 8] #fixed
agents_passed_through = [[] for _ in range(num_vertices)]
#importance = [np.random.randint(1, 10) for _ in range(num_vertices)] # dynamic
edges = [(0, 5), (0, 1), (1, 7), (2, 5), (2, 3), (3, 8), (3, 9), (4, 6), (4, 7), (4, 9), (8, 9), (7, 8), (5, 6),
         (5, 0), (1, 0), (7, 1), (5, 2), (3, 2), (8, 3), (9, 3), (6, 4), (7, 4), (9, 4), (9, 8), (8, 7), (6, 5)]
mycolor = ["blue","green","purple","black"]
for edge in edges:
    i, j = edge
    graph[j, i] = 1
    graph[i, j] = 1 


# Step 2: Initialize Agents
num_agents = 4
agent_positions = [0, 9, 3, 7]  # fixed 
#agent_positions = [np.random.randint(0, num_vertices) for _ in range(num_agents)] # dynamic
final_value = np.zeros(num_vertices)
for val in agent_positions : 
    idleness[val] -= 5
communication_range = 2


# Step 6: Visualization
def visualize_graph(agent_positions, ct, ax):
    ax.set_title(f"Graph Visualization of {ct}")

    # Draw edges
    for edge in edges:
        i, j = edge
        x = [i % 5, j % 5]
        y = [i // 5, j // 5]
        ax.plot(x, y, "b-", linewidth=1)

    # Draw nodes
    for i in range(num_vertices):
        x = i % 5
        y = i // 5
        ax.scatter(x, y, s=100, color="red", label=f"Node {i}")

    # Draw agents
    for i, pos in enumerate(agent_positions):
        x = pos % 5
        y = pos // 5
        ax.scatter(x, y, s=100, color=mycolor[i], label=f"Agent {i}")

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    

# To calculate the idleness of each vertex and the whole graph at time_step
def vertex_idleness(node, time_step):
    mean_idleness[node] = mean_idleness[node] + ( idleness[node] - mean_idleness[node])/time_step
    return mean_idleness[node]
def graph_idleness():
    total_idleness = sum(mean_idleness)
    num_vertices = len(idleness)
    return total_idleness / num_vertices

# Travel cost defined in this problem is the absolute difference between the value of nodes and agent position 
# We can define it to be a unequal weighted graph
def travel_cost(node,agent_positions):
    if graph[node,agent_positions]==1 : 
        return 1 
    return abs(node-agent_positions)

# Step 3: Define Utility Computation Function
def compute_utility(node, agent_positions):
    utility = importance[node] * idleness[node] - travel_cost(node,agent_positions)
    return utility

# The actual function 
def start(ct): 
    print(f"\n This is the {ct} iteration ") 
    time_step = ct + 1
    
    # Step 4: Phase 1 - Utility Computation
    utilities = []
    for agent in range(num_agents):
        current_position = agent_positions[agent]
        for node in range(num_vertices):
            if graph[current_position, node] == 1:
                utilities.append([node, agent, compute_utility(node, current_position)])          
    sorted_utilities = sorted(utilities, key=lambda x: x[2], reverse=True)      
    
    avg_graph_idleness = graph_idleness()
    print(f"Average Graph Idleness at iteration {ct}: {avg_graph_idleness}")

    # Calculate and print the vertex idleness for each vertex at this time step
    for vertex in range(num_vertices):
        avg_vertex_idleness = vertex_idleness(vertex, time_step)
        print(f"Vertex {vertex} avg Idleness : {avg_vertex_idleness} with idleness of {idleness[vertex]}")
  
  
    # Step 5: Phase 2 - Assignment with Consensus
    assignments = []
    assigned_nodes = set()
    assigned_agents = set()
    for i in range(len(utilities)):
        node_i, agent_i, utility_i = sorted_utilities[i] 
        if node_i not in assigned_nodes and agent_i not in assigned_agents:
            assigned_agents.add(agent_i)
            assigned_nodes.add(node_i)
            found_assignment = False
            for j in range(i+1, len(utilities)):
                node_j, agent_j, utility_j = sorted_utilities[j]
                if node_j not in assigned_nodes and agent_j not in assigned_agents:
                    if (utility_j == utility_i and node_i == node_j and 
                        abs(agent_positions[agent_i] - agent_positions[agent_j]) <= communication_range):
                        assigned_agents.add(agent_j)  
                        for k in range(j+1,len(utilities)):    
                            node_k, agent_k, utility_k = sorted_utilities[k]
                            if node_k not in assigned_nodes and agent_k not in assigned_agents:
                                if node_k != node_i and agent_k == agent_j:
                                    assignments.append([node_k, agent_j, utility_k])
                                    assignments.append([node_i, agent_i, utility_i])
                                    assigned_nodes.add(node_k)
                                    break
                                if node_k != node_i and agent_k == agent_i:
                                    assignments.append([node_k, agent_i, utility_k])
                                    assignments.append([node_j, agent_j, utility_j])
                                    assigned_nodes.add(node_k)
                                    break   
                        found_assignment = True
                        break
                    if utility_j < utility_i :
                        break    
            if not found_assignment:
                assignments.append([node_i, agent_i, utility_i])            
                        
    visited = [False]*num_vertices
    for node , agent , utility in assignments:
        print(f"Agent {agent} assigned actions: Node {node}")    
        agents_passed_through[node].append(agent)
        agent_positions[agent]=node 
        final_value[node] += 1 
        visited[node] = True 
        idleness[node] -= 5 
    
    # Updating the idleness of nodes where agents didnt go to
    for i in range(num_vertices): 
        if not visited[i]: 
            agents_passed_through[i].append('NULL')
            idleness[i] += 5
       
    print("\n\n")
    
for i in range(20):
    fig, ax = plt.subplots()
    start(i)
    visualize_graph(agent_positions, i, ax)
    plt.tight_layout()

plt.show()

for i in range(len(final_value)):
    print(f"{i} is visited {final_value[i]} and importance is {importance[i]}")
a = 0 
print("\n\n")
for inner_list in agents_passed_through:
    print(f" node {a} {inner_list} \n " )
    a = a+1 
    
