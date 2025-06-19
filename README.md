import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==============================================================================
# 1. HTML CONTENT AND FILE GENERATION
# ==============================================================================

# The HTML content describing the conceptual framework.
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Composite Neural Architecture</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
        }
        header {
            background: #333;
            color: #fff;
            padding: 10px 0;
            text-align: center;
        }
        section {
            margin: 20px;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .code-block {
            background: #272822;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .dot-cloud {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #0077ff;
        }
    </style>
</head>
<body>
    <header>
        <h1>Composite Neural Architecture: Hybrid Dot Cloud</h1>
    </header>
    <section>
        <h2>1. Introduction</h2>
        <p>This framework integrates diverse sensory and cognitive processes, such as vision, hearing, speech, and calculation, into a unified neural network structure called the <strong>Neural Iterative Architecture (NIA)</strong>.</p>
    </section>
    <section>
        <h2>2. Key Components</h2>
        <h3>2.1 Neural Iterative Architecture (NIA)</h3>
        <p>A multi-layered architecture with iterative refinement and feedback loops, ensuring dynamic adaptability.</p>
        <h3>2.2 Neuralion</h3>
        <p>An interlinked network combining specialized sub-networks for each modality. Connections are mediated through hyperpoints for cross-sensory integration.</p>
        <h3>2.3 Hyperpoints</h3>
        <p>Central nodes representing shared states in a multidimensional semantic space.</p>
    </section>
    <section>
        <h2>3. Workflow</h2>
        <ol>
            <li>Data streams enter specialized sub-networks for processing.</li>
            <li>Feature vectors are transformed into hyperpoints in a composite dot cloud.</li>
            <li>Hyperpoints are iteratively refined through feedback loops.</li>
            <li>Refined states are broadcast globally for decision-making.</li>
        </ol>
    </section>
    <section>
        <h2>4. Visualization</h2>
        <div class="dot-cloud">
            <div class="dot" style="background: #ff0000;"></div>
            <div class="dot" style="background: #00ff00;"></div>
            <div class="dot" style="background: #0000ff;"></div>
            <div class="dot" style="background: #ff00ff;"></div>
            <div class="dot" style="background: #00ffff;"></div>
            <div class="dot" style="background: #ffff00;"></div>
        </div>
        <p>Each dot represents a hyperpoint, with colors indicating modality (e.g., red for vision, green for hearing).</p>
    </section>
    <section>
        <h2>5. Applications</h2>
        <ul>
            <li><strong>Autonomous Systems:</strong> Robots capable of real-time decision-making based on multimodal input.</li>
            <li><strong>Creative AI:</strong> Generating novel art and music by combining sensory and cognitive data.</li>
            <li><strong>Healthcare:</strong> Integrative diagnostic tools blending patient history, imaging, and real-time monitoring.</li>
        </ul>
    </section>
    <footer>
        <p style="text-align:center;">© 2025 Composite Neural Architecture Framework</p>
    </footer>
</body>
</html>
"""

def create_html_file():
    """Writes the HTML content to a file."""
    try:
        with open('composite_neural_architecture.html', 'w', encoding='utf-8') as file:
            file.write(html_content)
        print("Success: HTML content written to 'composite_neural_architecture.html'")
    except IOError as e:
        print(f"Error writing to file: {e}")

# ==============================================================================
# 2. PYTHON CLASS DEFINITIONS FOR THE ARCHITECTURE
# ==============================================================================

class Neuron:
    """Represents a single neuron with dynamics modeled by a third-order differential equation."""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def third_order_equation(self, V, dV, ddV, I):
        """Calculates the neuron's state change based on its inputs."""
        return ddV + self.a * dV + self.b * V + self.c - I

class CompositePointCloud:
    """Represents a collection of data points (hyperpoints) in a 3D space, each with attributes."""
    def __init__(self):
        self.points = []

    def add_point(self, x, y, z, attributes):
        """Adds a point to the cloud."""
        self.points.append((x, y, z, attributes))

    def visualize(self):
        """Creates a 3D scatter plot of the point cloud."""
        if not self.points:
            print("Point cloud is empty. Nothing to visualize.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs, _ = zip(*self.points)
        ax.scatter(xs, ys, zs, c='b', marker='o')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('Composite Point Cloud Visualization')
        plt.show()

class RandomHybridTree:
    """Represents a directed graph structure for modeling relationships between nodes."""
    def __init__(self):
        self.tree = nx.DiGraph()

    def add_node(self, node):
        """Adds a node to the graph."""
        self.tree.add_node(node)

    def add_edge(self, node1, node2, weight):
        """Adds a weighted, directed edge between two nodes."""
        self.tree.add_edge(node1, node2, weight=weight)

    def visualize(self):
        """Draws the graph."""
        if not self.tree.nodes:
            print("Tree is empty. Nothing to visualize.")
            return
            
        pos = nx.spring_layout(self.tree)
        edge_labels = nx.get_edge_attributes(self.tree, 'weight')
        
        plt.figure()
        nx.draw(self.tree, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(self.tree, pos, edge_labels=edge_labels)
        plt.title('Random Hybrid Tree Visualization')
        plt.show()

class CompositeNeuralArchitecture:
    """An integrated architecture combining neurons, a point cloud, and a hybrid tree."""
    def __init__(self):
        self.neurons = []
        self.point_cloud = CompositePointCloud()
        self.hybrid_tree = RandomHybridTree()

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_point_to_cloud(self, x, y, z, attributes):
        self.point_cloud.add_point(x, y, z, attributes)

    def add_edge_to_tree(self, node1, node2, weight):
        # Ensure nodes exist before adding an edge
        if node1 not in self.hybrid_tree.tree:
            self.hybrid_tree.add_node(node1)
        if node2 not in self.hybrid_tree.tree:
            self.hybrid_tree.add_node(node2)
        self.hybrid_tree.add_edge(node1, node2, weight)

    def visualize_architecture(self):
        """Visualizes all components of the architecture."""
        print("\nVisualizing Composite Point Cloud from Architecture...")
        self.point_cloud.visualize()
        print("Visualizing Random Hybrid Tree from Architecture...")
        self.hybrid_tree.visualize()


# ==============================================================================
# 3. SCRIPT EXECUTION AND DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    # Part 1: Generate the HTML documentation file
    create_html_file()
    print("-" * 50)
    
    # Part 2: Demonstrate the core components individually
    print("--- Individual Component Demonstration ---")

    # Neuron dynamics using third-order differential equations
    neuron = Neuron(a=1.0, b=0.5, c=-0.1)
    V, dV, ddV, I = 1.0, 0.5, 0.2, 0.8
    neuron_output = neuron.third_order_equation(V, dV, ddV, I)
    print(f"Neuron output (third-order equation): {neuron_output:.2f}")

    # Create and visualize a composite point cloud
    print("\nVisualizing an individual Point Cloud...")
    point_cloud = CompositePointCloud()
    point_cloud.add_point(0.1, 0.2, 0.3, {"firing_rate": 0.5})
    point_cloud.add_point(0.4, 0.5, 0.6, {"firing_rate": 0.8})
    point_cloud.add_point(-0.2, 0.7, -0.1, {"firing_rate": 0.3})
    point_cloud.visualize()

    # Create and visualize a random hybrid tree
    print("Visualizing an individual Hybrid Tree...")
    hybrid_tree = RandomHybridTree()
    hybrid_tree.add_node("A")
    hybrid_tree.add_node("B")
    hybrid_tree.add_edge("A", "B", 0.6)
    hybrid_tree.visualize()
    print("-" * 50)

    # Part 3: Demonstrate the integrated architecture
    print("\n--- Integrated Architecture Demonstration ---")
    
    # Initialize the complete architecture
    architecture = CompositeNeuralArchitecture()
    
    # Add components to the architecture
    architecture.add_neuron(neuron)
    architecture.add_point_to_cloud(0.7, 0.8, 0.9, {"modality": "vision"})
    architecture.add_point_to_cloud(-0.5, -0.4, 0.2, {"modality": "audio"})
    architecture.add_edge_to_tree("Vision_Input", "Hyperpoint_A", 0.9)
    architecture.add_edge_to_tree("Audio_Input", "Hyperpoint_B", 0.85)
    architecture.add_edge_to_tree("Hyperpoint_A", "Decision_Node", 0.7)
    architecture.add_edge_to_tree("Hyperpoint_B", "Decision_Node", 0.6)

    # Visualize the components as managed by the main architecture object
    architecture.visualize_architecture()
    
    print("\nDemonstration complete.")