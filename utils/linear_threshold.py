import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from matplotlib.animation import FuncAnimation
from joblib import Parallel, delayed

class LinearThreshold:
    def trial(self, n, all):
        """Simulate a contagion spreading along a network with the linear threshold model

        Args:
        n (int): number of nodes in the network
        all (bool): True if the initial contagion must start in the largest component.

        Returns:
        tuple: a tuple contaning containing the following information
          infected: The number of infected individuals
          N: number of nodes in largest component if all = True; otherwise return n
          z: expected degree of a node in the network
        """

        # Network topology
        z = np.random.uniform(2,7)
        p = z/n
        graph = nx.erdos_renyi_graph(n, p)

        if not all:
            G_components = sorted(nx.connected_components(graph), key=len, reverse=True)
            g = graph.subgraph(G_components[0])
            N = g.number_of_nodes()
        else:
            g = graph
            N = n

        # Model selection
        model = ep.ThresholdModel(g)

        # Model Configuration
        config = mc.Configuration()
        config.add_model_parameter('fraction_infected', 2/N)

        # Setting node parameters
        m = 2*z/(z-1+(1-p)**N)
        thres_list = np.random.uniform(0, 1/m, size = n)
        for i in range(n):
            config.add_node_configuration("threshold", i, thres_list[i])

        model.set_initial_status(config)

        # Simulation execution
        iterations = model.iteration_bunch(50)

        trends = model.build_trends(iterations)
        infected = trends[0]["trends"]["node_count"][1][-1]

        return infected, N, z
    
    def multi_trials_worker(self, n, all, num_trials = 10):
        """Simulate multiple trajectories of contagion spreading along a network with the linear threshold model using a single worker

        Args:
        n (int): number of nodes in the network
        all (bool): True if the initial contagion must start in the largest component.
        num_trials (int): number of trajectories

        Returns:
        list: a list of tuples contaning containing the following information
          infected: The number of infected individuals
          N: number of nodes in largest component if all = True; otherwise return n
          z: expected degree of a node in the network
        """
        
        results = []
        for i in range(num_trials):
            infected, N, z = self.trial(n, all)
            results.append((infected, N, z))
        return results

    def multi_trials(self, n, all, num_trials = 10, n_cpu = 2):
        """Simulate multiple trajectories of contagion spreading along a network with the linear threshold model using multiple workers

        Args:
        n (int): number of nodes in the network
        all (bool): True if the initial contagion must start in the largest component.
        num_trials (int): number of trajectories
        n_cpu (int): number of CPUs to use

        Returns:
        list: a list of tuples contaning containing the following information
          infected: The number of infected individuals
          N: number of nodes in largest component if all = True; otherwise return n
          z: expected degree of a node in the network
        """
        
        batch_size = int(math.ceil(num_trials / n_cpu))
        results = Parallel(n_jobs = n_cpu)(delayed(self.multi_trials_worker)(
            n, all, min((i + 1) * batch_size, num_trials) - i * batch_size
        ) for i in range(n_cpu))
        ans = []
        for res in results:
            ans += res
        return ans
    
    def draw_graph(self, i, g, iterations, cent, node_pos, snapshot, thres):
        """A helper function for animate(). Draws graph given a single snapshot

        Args:
        i (int): the current iteration in the animation
        g (Networkx.Graph): a networkx graph module
        iterations (list): a list of network infection status for all iterations
        cent (dict): the dictionary of edges with betweeness centrality for plotting
        node_pos (dict): the position of nodes on the plot
        snapshot (dict): a snapshot of infection status
        thres (float): the upper bound of linear threshold
        """
        
        plt.clf()
        color_map = []
        dct = iterations[i]["status"]
        for key in dct:
            snapshot[key] = dct[key]
        num_nodes = 0
        infected_nodes = 0
        for node in g:
            if node in snapshot and snapshot[node] == 1:
                color_map.append("#ffc3d7")
                infected_nodes += 1
            else:
                color_map.append("#badaff")
            num_nodes += 1
        frac_infected = infected_nodes / num_nodes
        nx.draw_networkx_nodes(g, pos = node_pos, node_color = color_map, node_size = 50)
        nx.draw_networkx_edges(g, pos = node_pos, edgelist = cent, alpha = 0.1)
        plt.title(f"Threshold: {thres} - Iteration #{i+1} Infected {(frac_infected * 100):.2f}%")
#        if i in [0, 4, 9]:
#            plt.savefig(f"plots/itr={i}.png")

    def animate(self, thres_up = 0.5, n = 100):
        """Animate the infection transmission in a network

        Args:
        thres_up (float): the upper bound of linear threshold
        n (int): number of nodes in the network
        """
        
        # Model Configuration
        config = mc.Configuration()
        config.add_model_parameter("fraction_infected", 0.02)
        # Setting node parameters
        thres_lst = np.random.uniform(0, thres_up, size = n)
        for i in range(n):
            config.add_node_configuration("threshold", i, thres_lst[i])
        # Network topology
        z = np.random.uniform(2,7)
        p = z/n
        g = nx.erdos_renyi_graph(n, p)
        largest_cc = max(nx.connected_components(g), key = len)
        # Model selection
        model = ep.ThresholdModel(g)
        model.set_initial_status(config)
        # Simulation execution
        iterations = model.iteration_bunch(30)
        trends = model.build_trends(iterations)
        snapshot = {}
        num_infected = 0
        num_nodes = 0
        for i in range(30):
            dct = iterations[i]["status"]
            for key in dct:
                snapshot[key] = dct[key]
        for key in snapshot:
            if key in largest_cc:
                num_infected += snapshot[key]
                num_nodes += 1
        ## Animation
        largest_cc = max(nx.connected_components(g), key = len)
        g = g.subgraph(largest_cc)
        cent=nx.edge_betweenness_centrality(g)
        node_pos=nx.spring_layout(g)
        snapshot = {}
        fig = plt.figure()
        anim = FuncAnimation(fig, self.draw_graph, frames = len(iterations), interval = 1000, fargs=(g, iterations, cent, node_pos, snapshot, thres_up), repeat = True)
        plt.show()
