import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameter w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """

    # Number of training examples (length of x)
    m = x.shape[0]    

    # Initialize gradients for w and b
    dj_dw = 0
    dj_db = 0
    
    # Loop over each training example
    for i in range(m):  
        # Compute the model prediction for example i: f_wb = w*x + b
        f_wb = w * x[i] + b 

        # Contribution of example i to gradient w.r.t w (calculate the derivative of the cost function with respect to w)
        dj_dw_i = (f_wb - y[i]) * x[i] 

        # Contribution of example i to gradient w.r.t b (calculate the derivative of the cost function with respect to b)   
        dj_db_i = f_wb - y[i] 

        # Accumulate the contributions into total gradients
        dj_db += dj_db_i
        dj_dw += dj_dw_i 

    # Average the gradients over all training examples
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    # Return the gradients
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """
    
    # Initialize lists to store cost values and parameter history for visualization
    J_history = []   # List to store cost at each iteration
    p_history = []   # List to store [w, b] values at each iteration
    
    # Initialize parameters with input values
    b = b_in        # Current value of bias
    w = w_in        # Current value of weight
    
    # Loop over the number of iterations for gradient descent
    for i in range(num_iters):
        # Compute the gradient of cost with respect to w and b using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update bias parameter using gradient descent rule
        b = b - alpha * dj_db                            
        # Update weight parameter using gradient descent rule
        w = w - alpha * dj_dw                            

        # Store cost and parameters for graphing, avoid excessive memory usage
        if i < 100000:      
            J_history.append(cost_function(x, y, w , b))   # Compute and save current cost
            p_history.append([w, b])                        # Save current parameters
        
        # Print progress every 10% of total iterations
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    # Return final parameters and history of cost and parameters
    return w, b, J_history, p_history

def plt_intuition(x_train, y_train):
    """
    Plot training data and cost function J for the given data
    """
    # Create a range of w values for the cost function
    w_range = np.linspace(0, 400, 1000)  # Changed range to match your graph
    b = 100  # Changed from 0 to 100 to match your graph
    
    # Calculate cost function J for different w values using compute_cost
    J_values = []
    for w in w_range:
        J = compute_cost(x_train, y_train, w, b)
        J_values.append(J)
    
    J_values = np.array(J_values) # keeping all the J values in a np array for plotting
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training data
    ax1.scatter(x_train, y_train, marker='x', c='r', s=100, label='Training Data')
    ax1.set_xlabel('Size (1000 sqft)')
    ax1.set_ylabel('Price (1000s)')
    ax1.set_title('Housing Data')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Cost function J - Updated to match your graph
    ax2.plot(w_range, J_values, 'b-', linewidth=3, label='Cost Function J')
    ax2.set_xlabel('w')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost vs. w, (b fixed at 100)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set axis limits to match your graph
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 50000)
    
    # Find and mark the minimum
    min_idx = np.argmin(J_values)
    min_w = w_range[min_idx]
    min_J = J_values[min_idx]
    ax2.plot(min_w, min_J, 'ro', markersize=10, label=f'cost at w={min_w:.0f}')
    
    # Add reference lines like in your graph
    ax2.axhline(y=0, color='purple', linestyle='--', alpha=0.7)
    ax2.axvline(x=min_w, ymin=0, ymax=min_J/50000, color='purple', linestyle='--', alpha=0.7)
    
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Minimum cost J = {min_J:.2f} at w = {min_w:.0f}")
    print(f"Your data: x_train = {x_train}, y_train = {y_train}")
    print(f"Model: Price = {min_w:.0f} × Size + {b}")

def soup_bowl(x_train, y_train):
    """
    Creates a 3D 'soup bowl' visualization of the cost function J(w,b)
    """
    # Define ranges for w and b
    w_values = np.linspace(0, 400, 50)
    b_values = np.linspace(-100, 400, 50)
    
    # Create meshgrid
    W, B = np.meshgrid(w_values, b_values)
    
    # Calculate J(w,b) for each (w,b) pair
    J_surface = np.zeros_like(W)
    for i in range(len(w_values)):
        for j in range(len(b_values)):
            J_surface[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface (the "soup bowl")
    surf = ax.plot_surface(W, B, J_surface, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add contour lines on the bottom
    ax.contour(W, B, J_surface, zdir='z', offset=J_surface.min(), 
               cmap='viridis', alpha=0.5, levels=10)
    
    # Find and mark the minimum point
    min_idx = np.unravel_index(np.argmin(J_surface), J_surface.shape)
    min_w = W[min_idx]
    min_b = B[min_idx]
    min_J = J_surface[min_idx]
    
    # Mark the minimum with a red dot
    ax.scatter(min_w, min_b, min_J, color='red', s=100, 
               label=f'Minimum: w={min_w:.0f}, b={min_b:.0f}, J={min_J:.0f}')
    
    # Set labels and title
    ax.set_xlabel('w (weight)')
    ax.set_ylabel('b (bias)')
    ax.set_zlabel('J(w,b)')
    ax.set_title('Cost Function J(w,b) - "Soup Bowl"')
    ax.legend()
    
    # Add some styling
    ax.view_init(elev=20, azim=45)  # Set viewing angle
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Minimum cost J = {min_J:.2f} at w = {min_w:.0f}, b = {min_b:.0f}")

def plt_gradients(x_train, y_train, compute_cost, compute_gradient):
    """
    Plots the cost function and gradients for linear regression
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left plot: Cost vs w with gradient
    w_range = np.linspace(0, 400, 100)
    b_fixed = 100
    costs = [compute_cost(x_train, y_train, w, b_fixed) for w in w_range]
    
    ax1.plot(w_range, costs, 'lightblue', linewidth=2, label='Cost function')
    ax1.set_xlabel('w')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost vs w, with gradient; b set to 100')
    ax1.grid(True, alpha=0.3)
    
    # Add gradient points and tangent lines
    w_points = [100, 200, 300]
    colors = ['blue', 'green', 'red']
    
    for i, w_val in enumerate(w_points):
        cost_val = compute_cost(x_train, y_train, w_val, b_fixed)
        dj_dw, dj_db = compute_gradient(x_train, y_train, w_val, b_fixed)
        
        # Plot point
        ax1.plot(w_val, cost_val, 'o', color=colors[i], markersize=8)
        
        # Plot tangent line (gradient)
        w_tangent = np.linspace(w_val - 50, w_val + 50, 10)
        cost_tangent = cost_val + dj_dw * (w_tangent - w_val)
        ax1.plot(w_tangent, cost_tangent, '--', color='red', alpha=0.7, linewidth=2)
        
        # Add gradient annotation
        ax1.annotate(f'∂J/∂w = {dj_dw:.0f}', 
                    xy=(w_val, cost_val), 
                    xytext=(10, 20), 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Right plot: Quiver plot of gradients
    w_range_2d = np.linspace(-100, 600, 20)
    b_range_2d = np.linspace(-200, 200, 20)
    W, B = np.meshgrid(w_range_2d, b_range_2d)
    
    # Calculate gradients at each point
    dj_dw_grid = np.zeros_like(W)
    dj_db_grid = np.zeros_like(B)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            dj_dw_grid[i, j], dj_db_grid[i, j] = compute_gradient(x_train, y_train, W[i, j], B[i, j])
    
    # Create quiver plot
    ax2.quiver(W, B, -dj_dw_grid, -dj_db_grid, 
               np.sqrt(dj_dw_grid**2 + dj_db_grid**2), 
               cmap='viridis', alpha=0.7, scale=1000)
    ax2.set_xlabel('w')
    ax2.set_ylabel('b')
    ax2.set_title('Gradient shown in quiver plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plt_contour_wgrad(x_train, y_train, p_history, ax):
    """
    Plots the contour plot of cost function with gradient descent path
    """
    # Create a grid of w and b values
    w_range = np.linspace(-100, 400, 50)
    b_range = np.linspace(-400, 400, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    # Calculate cost for each point in the grid
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])
    
    # Create contour plot
    contour_levels = [1000, 5000, 10000, 25000, 50000]
    CS = ax.contour(W, B, Z, levels=contour_levels, colors=['darkred', 'magenta', 'purple', 'blue', 'orange'], linewidths=2)
    ax.clabel(CS, inline=True, fontsize=10, fmt='%d')
    
    # Plot gradient descent path
    if len(p_history) > 0:
        p_hist = np.array(p_history)
        w_path = p_hist[:, 0]
        b_path = p_hist[:, 1]
        
        # Plot the path with arrows
        ax.plot(w_path, b_path, 'r-', linewidth=3, alpha=0.8)
        
        # Add arrows to show direction
        for i in range(0, len(w_path)-1, max(1, len(w_path)//20)):
            dx = w_path[i+1] - w_path[i]
            dy = b_path[i+1] - b_path[i]
            ax.arrow(w_path[i], b_path[i], dx, dy, 
                    head_width=5, head_length=5, fc='red', ec='red', alpha=0.7)
        
        # Add cost annotations at key points
        if len(p_history) > 0:
            # First point
            cost_0 = compute_cost(x_train, y_train, w_path[0], b_path[0])
            ax.annotate(f'{cost_0:.0f}', xy=(w_path[0], b_path[0]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Middle point
            mid_idx = len(w_path) // 2
            cost_mid = compute_cost(x_train, y_train, w_path[mid_idx], b_path[mid_idx])
            ax.annotate(f'{cost_mid:.0f}', xy=(w_path[mid_idx], b_path[mid_idx]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Final point
            cost_final = compute_cost(x_train, y_train, w_path[-1], b_path[-1])
            ax.annotate(f'{cost_final:.0f}', xy=(w_path[-1], b_path[-1]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add reference lines to minimum
    ax.axvline(x=200, color='purple', linestyle=':', alpha=0.7)
    ax.axhline(y=100, color='purple', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('Contour plot of cost J(w,b), vs b,w with path of gradient descent')
    ax.grid(True, alpha=0.3)

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

def create_restaurant_dataset(n_samples=100, random_seed=42):
    """
    Creates a fake dataset for restaurant franchise problem
    
    Args:
        n_samples (int): Number of cities to simulate
        random_seed (int): Random seed for reproducibility
    
    Returns:
        x_train (ndarray): City populations in thousands
        y_train (ndarray): Restaurant profits in thousands of dollars
    """
    np.random.seed(random_seed)
    
    # Generate city populations (in thousands)
    # Realistic range: small towns (10k) to large cities (1000k)
    #x_train = np.random.uniform(10, 1000, n_samples)
    
    # Add some realistic distribution - more small cities, fewer large ones
    # Numpy exponential is a function that generates random numbers 
    # from an exponential distribution
    # scale is the scale of the exponential distribution
    # size is the number of samples to generate
    x_train = np.random.exponential(scale=50, size=n_samples) + 10
    # clip the values to be between 10 and 1000
    x_train = np.clip(x_train, 10, 1000)  # Cap at reasonable limits
    
    # Generate profits based on population with some realistic factors
    # Basic relationship: profit increases with population
    # But with diminishing returns and some randomness
    
    # Base profit from population (diminishing returns)
    base_profit = 0.5 * x_train - 0.0002 * x_train**2 + 50
    
    # Add random noise (competition, local factors, etc.)
    noise = np.random.normal(0, 30, n_samples)
    
    # Some cities might be unprofitable (negative profit)
    y_train = base_profit + noise
    
    # Ensure some negative profits for realism
    # About 20% of restaurants might be unprofitable
    negative_mask = np.random.random(n_samples) < 0.2
    y_train[negative_mask] = np.random.uniform(-50, -10, np.sum(negative_mask))
    
    return x_train, y_train

def load_data():
    """
    Load the restaurant dataset - returns fake data for practice
    """
    return create_restaurant_dataset(n_samples=100, random_seed=42)

def print_data(x, y, num_examples=10):
    """
    Prints the dataset in a nice format
    
    Args:
        x (ndarray): Input features (city populations)
        y (ndarray): Target values (restaurant profits)
        num_examples (int): Number of examples to print
    """
    print(f"Dataset contains {len(x)} examples")
    print("-" * 30) # 30 is the length of the line
    print("Population(k) | Profit(k$)")
    print("-" * 30)
    
    for i in range(min(num_examples, len(x))):
        print(f"{x[i]:10.1f} | {y[i]:8.1f}")
    
    if len(x) > num_examples:
        print(f"... and {len(x) - num_examples} more examples")
    
    print("-" * 30)
    print(f"Population range: {x.min():.1f}k to {x.max():.1f}k")
    print(f"Profit range: ${y.min():.1f}k to ${y.max():.1f}k")
