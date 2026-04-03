import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from swarm_manager import SwarmManager

def main():
    dt = 0.1
    v_max = 5.0
    u_max = 2.0
    drones_count = 4

    fake_vertices = [[0, 0], [100, 0], [100, 100], [0, 100]]
    
    fake_targets = [[50, 0], [100, 50], [50, 100], [0, 50]]

    initial_positions = [
        [random.randint(-20, 120), random.randint(-20, 120)],
        [random.randint(-20, 120), random.randint(-20, 120)],
        [random.randint(-20, 120), random.randint(-20, 120)],
        [random.randint(-20, 120), random.randint(-20, 120)]
    ]
    
    swarm = SwarmManager(drones_count, initial_positions, v_max, u_max)

    # Set up the visual canvas
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-30, 130)
    ax.set_ylim(-30, 130)
    ax.set_aspect('equal') # Ensures the square doesn't get stretched into a rectangle
    
    # 1. Draw the Extended Hull
    # We add the first vertex to the end of the lists so the line closes the loop
    poly_x = [v[0] for v in fake_vertices] + [fake_vertices[0][0]]
    poly_y = [v[1] for v in fake_vertices] + [fake_vertices[0][1]]
    ax.plot(poly_x, poly_y, 'b-', label='Extended Hull')
    
    # 2. Draw the Target Steering Points
    target_x = [t[0] for t in fake_targets]
    target_y = [t[1] for t in fake_targets]
    ax.scatter(target_x, target_y, color='green', marker='*', s=150, label='Targets')
    
    # 3. Prepare the Drone Graphics
    # We initialize an empty scatter plot that we will update every frame
    drone_scatter = ax.scatter([], [], color='red', s=50, zorder=5, label='Drones')
    ax.legend()

    def animate(frame):
        """
        This function runs every time the screen refreshes.
        """
        # Step 1: Run your math to calculate allocations and control laws
        swarm.update_swarm(dt, fake_vertices, fake_targets)
        
        # Step 2: Retrieve the physical result
        positions, headings = swarm.get_swarm_status()
        
        # Step 3: Update the red dots on the screen
        drone_scatter.set_offsets(positions)
        
        # Return the graphic object so matplotlib knows what to redraw
        return drone_scatter,

    # Set up the animation engine
    # frames=200 means the simulation will run for 200 time steps
    # interval=50 means the screen updates every 50 milliseconds
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
    
    # Open the window and start the simulation!
    plt.show()

if __name__ == "__main__":
    main()