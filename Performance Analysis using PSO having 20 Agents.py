#Harmehak Singh Khangura
import pygame
import numpy as np
import sys
import math
from collections import deque
import pandas as pd  
import matplotlib.pyplot as plt  


Screen_width = 800
Screen_height = 600
Grid_size = 40

Cells_X = Screen_width // Grid_size
Cells_Y = Screen_height // Grid_size

Agent_size = 10
Target_size = 15
Neighborhood_radius = 100  

White = (255, 255, 255)
Light_gray = (200, 200, 200)
Gray = (100, 100, 100)
Dark_gray = (80, 80, 80)
Black = (0, 0, 0)
Blue = (0, 0, 255)
Green = (0, 200, 0)
Red = (255, 0, 0)


W = 0.5         
C1 = 1.5        
C2 = 1.5        
Max_speed = 2   

def run_simulation(Agents=20, Targets=5, Obstacles=50, display=False):
    pygame.init()

    obstacle_grid = np.zeros((Cells_X, Cells_Y), dtype=bool)

    for _ in range(Obstacles):
        ox, oy = np.random.randint(0, Cells_X), np.random.randint(0, Cells_Y)
        obstacle_grid[ox, oy] = True

    explored_cells = np.zeros((Cells_X, Cells_Y))

    class Agent:
        def __init__(self, x, y, targets):
            self.x = x
            self.y = y
            self.vx = np.random.uniform(-Max_speed, Max_speed)
            self.vy = np.random.uniform(-Max_speed, Max_speed)
            self.best_x = x
            self.best_y = y
            self.best_distance = float('inf')
            self.found_targets = []
            self.distance_traveled = 0
            self.visited_cells = set()
            self.is_idle = False
            self.idle_time = 0
            self.last_state_change_time = pygame.time.get_ticks()
            self.neighbors = []
            self.targets = targets
            self.has_found_target = False

        def update_velocity(self):
            local_best_x, local_best_y = self.x, self.y
            local_best_distance = self.best_distance

            for neighbor in self.neighbors:
                if neighbor.best_distance < local_best_distance and neighbor != self:
                    local_best_distance = neighbor.best_distance
                    local_best_x = neighbor.best_x
                    local_best_y = neighbor.best_y

            r1 = np.random.rand()
            r2 = np.random.rand()

            cognitive_x = C1 * r1 * (self.best_x - self.x)
            social_x = C2 * r2 * (local_best_x - self.x)
            self.vx = W * self.vx + cognitive_x + social_x + np.random.uniform(-2, 2)

            cognitive_y = C1 * r1 * (self.best_y - self.y)
            social_y = C2 * r2 * (local_best_y - self.y)
            self.vy = W * self.vy + cognitive_y + social_y + np.random.uniform(-2, 2)

            speed = math.hypot(self.vx, self.vy)
            if speed > Max_speed:
                self.vx = (self.vx / speed) * Max_speed
                self.vy = (self.vy / speed) * Max_speed

        def avoid_obstacles(self):
            future_x = self.x + self.vx * 2
            future_y = self.y + self.vy * 2

           
            future_x = max(0, min(future_x, Screen_width - 1))
            future_y = max(0, min(future_y, Screen_height - 1))

            grid_x = int(future_x // Grid_size)
            grid_y = int(future_y // Grid_size)

           
            if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
                if obstacle_grid[grid_x, grid_y]:
                    angle = math.atan2(self.vy, self.vx)
                    avoid_angle = angle + np.pi / 2
                    self.vx = math.cos(avoid_angle) * Max_speed
                    self.vy = math.sin(avoid_angle) * Max_speed

        def move(self, agents):
            current_time = pygame.time.get_ticks()
            self.avoid_obstacles()

            new_x = self.x + self.vx
            new_y = self.y + self.vy

           
            new_x = max(0, min(new_x, Screen_width - 1))
            new_y = max(0, min(new_y, Screen_height - 1))

            grid_x = int(new_x // Grid_size)
            grid_y = int(new_y // Grid_size)

            collision_occurred = False 

            if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
                collision = False
                for agent in agents:
                    if agent != self:
                        if math.hypot(agent.x - new_x, agent.y - new_y) < Agent_size * 2:
                            collision = True
                            collision_occurred = True
                            break
                if not collision:
                    self.x = new_x
                    self.y = new_y
                    self.distance_traveled += math.hypot(self.vx, self.vy)
                    self.is_idle = False
                else:
                   
                    self.vx += np.random.uniform(-1, 1)
                    self.vy += np.random.uniform(-1, 1)
                    self.idle_time += current_time - self.last_state_change_time
                    self.last_state_change_time = current_time
            else:
               
                if self.x <= 0 or self.x >= Screen_width - 1:
                    self.vx = -self.vx
                if self.y <= 0 or self.y >= Screen_height - 1:
                    self.vy = -self.vy
                self.x = max(0, min(self.x, Screen_width - 1))
                self.y = max(0, min(self.y, Screen_height - 1))

           
            grid_x = int(self.x // Grid_size)
            grid_y = int(self.y // Grid_size)
            if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
                explored_cells[grid_x, grid_y] = 1

            return collision_occurred  

        def check_for_target(self):
            for target in self.targets[:]:
                distance = math.hypot(self.x - target.x, self.y - target.y)
                if distance < Target_size:
                    self.targets.remove(target)
                    self.found_targets.append(target)
                    self.has_found_target = False 
                    self.best_distance = float('inf')
                    break

            for target in self.targets:
                distance = math.hypot(self.x - target.x, self.y - target.y)
                if distance < self.best_distance:
                    self.best_x = target.x
                    self.best_y = target.y
                    self.best_distance = distance

        def update_neighbors(self, agents):
            self.neighbors = []
            for agent in agents:
                if agent != self:
                    distance = math.hypot(self.x - agent.x, self.y - agent.y)
                    if distance < Neighborhood_radius:
                        self.neighbors.append(agent)

        def draw(self, screen):
            if display:
                color = Green if self.has_found_target else Blue
                pygame.draw.circle(screen, color, (int(self.x), int(self.y)), Agent_size)

    class Target:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def draw(self, screen):
            if display:
                pygame.draw.circle(screen, Red, (int(self.x), int(self.y)), Target_size)

    def draw_grid(screen):
        if display:
            for x in range(0, Screen_width, Grid_size):
                pygame.draw.line(screen, Light_gray, (x, 0), (x, Screen_height))
            for y in range(0, Screen_height, Grid_size):
                pygame.draw.line(screen, Light_gray, (0, y), (Screen_width, y))

    def draw_obstacles(screen):
        if display:
            for i in range(Cells_X):
                for j in range(Cells_Y):
                    if obstacle_grid[i, j]:
                        pygame.draw.rect(
                            screen,
                            Dark_gray,
                            (i * Grid_size, j * Grid_size, Grid_size, Grid_size)
                        )

    def run():
        screen = pygame.display.set_mode((Screen_width, Screen_height)) if display else pygame.Surface((Screen_width, Screen_height))
        pygame.display.set_caption("Search and Rescue Simulation")

        font = pygame.font.SysFont(None, 24) if display else None

        agents = []
        targets = []

        collision_count = 0 

       
        while len(targets) < Targets:
            x = np.random.uniform(0, Screen_width - 1)
            y = np.random.uniform(0, Screen_height - 1)
            grid_x = int(x // Grid_size)
            grid_y = int(y // Grid_size)
            if not obstacle_grid[grid_x, grid_y]:
                targets.append(Target(x, y))

        while len(agents) < Agents:
            x = np.random.uniform(0, Screen_width - 1)
            y = np.random.uniform(0, Screen_height - 1)
            grid_x = int(x // Grid_size)
            grid_y = int(y // Grid_size)
            if not obstacle_grid[grid_x, grid_y]:
                agents.append(Agent(x, y, targets))

        start_time = pygame.time.get_ticks()
        running = True
        clock = pygame.time.Clock()
        paused = False

        while running:
            if display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            paused = not paused
                        if event.key == pygame.K_r:
                            run()
                            return

            if not paused:
                if display:
                    screen.fill(White)
                    draw_grid(screen)
                    draw_obstacles(screen)

                for agent in agents:
                    agent.update_neighbors(agents)

                for agent in agents:
                    agent.update_velocity()
                    collision_occurred = agent.move(agents)
                    if collision_occurred:
                        collision_count += 1 
                    agent.check_for_target()
                    agent.draw(screen)

                for target in targets:
                    target.draw(screen)

                if len(targets) == 0:
                    total_time = pygame.time.get_ticks() - start_time
                    distance_traveled = sum([agent.distance_traveled for agent in agents])

                    total_idle_time = sum([agent.idle_time for agent in agents])
                    idle_time = total_idle_time

                    exploration_coverage = np.sum(explored_cells) / (Cells_X * Cells_Y) * 100

                    if display:
                        print(f"All targets found in {total_time} ms.")
                        print(f"Total distance travelled: {distance_traveled}")
                        print(f"Exploration coverage: {exploration_coverage}%")
                        print(f"Idle time: {idle_time} ms")
                        print(f"Collisions: {collision_count}")

                    running = False
                    pygame.quit()
                    return {
                        'Agents': Agents,
                        'Targets': Targets,
                        'Obstacles': Obstacles,
                        'total_time': total_time,
                        'distance_traveled': distance_traveled,
                        'exploration_coverage': exploration_coverage,
                        'idle_time': idle_time,
                        'collision_count': collision_count,
                    }

                if display:
                    time_text = font.render(f"Time: {pygame.time.get_ticks() - start_time} ms", True, Black)
                    screen.blit(time_text, (10, 10))
                    target_text = font.render(f"Targets Remaining: {len(targets)}", True, Black)
                    screen.blit(target_text, (10, 30))
                    collision_text = font.render(f"Collisions: {collision_count}", True, Black)
                    screen.blit(collision_text, (10, 50))

                    instruction_text = font.render("Press 'P' to pause, 'R' to reset", True, Black)
                    screen.blit(instruction_text, (10, Screen_height - 30))

                    pygame.display.flip()
                    clock.tick(30)
                else:
                    clock.tick(30)
            else:
                if display:
                    pause_text = font.render("Paused. Press 'P' to resume.", True, Black)
                    screen.blit(pause_text, (Screen_width // 2 - 100, Screen_height // 2))
                    pygame.display.flip()
                    clock.tick(30)
                else:
                    clock.tick(30)

        pygame.quit()
        sys.exit()

   
    data = run()
    return data

def run_test_cases(num_runs):
    data_list = []
    for i in range(num_runs):
        print(f"Running simulation {i+1}/{num_runs}")

      
        Agents_count = 20
        Targets_count = 5
        Obstacles_count = 50

       
        data = run_simulation(Agents=Agents_count, Targets=Targets_count, Obstacles=Obstacles_count, display=False)
        data_list.append(data)

   
    df = pd.DataFrame(data_list)

   
    df.to_csv('simulation_results.csv', index=False)
    print("\nSimulation results saved to 'simulation_results.csv'.")

   
    metrics = ['total_time', 'distance_traveled', 'exploration_coverage', 'idle_time', 'collision_count']

    
    df_normalized = df.copy()
    for metric in metrics:
        min_value = df[metric].min()
        max_value = df[metric].max()
        if max_value - min_value != 0:
            df_normalized[metric] = (df[metric] - min_value) / (max_value - min_value)
        else:
            df_normalized[metric] = 0.0  

    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(range(1, num_runs + 1), df_normalized[metric], marker='o', linestyle='-', label=metric.replace('_', ' ').title())

    plt.title('All Metrics Over Simulation Runs (Normalized)')
    plt.xlabel('Simulation Run')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('all_metrics_normalized.png')
    plt.close() 

    print("\nAll metrics have been plotted in a single figure 'all_metrics_normalized.png'.")

   
    avg_total_time = df['total_time'].mean()
    avg_distance_traveled = df['distance_traveled'].mean()
    avg_exploration_coverage = df['exploration_coverage'].mean()
    avg_idle_time = df['idle_time'].mean()
    avg_collision_count = df['collision_count'].mean()

    print("\nAverage Results after {} runs:".format(num_runs))
    print(f"Average total time: {avg_total_time:.2f} ms")
    print(f"Average distance travelled: {avg_distance_traveled:.2f}")
    print(f"Average exploration coverage: {avg_exploration_coverage:.2f}%")
    print(f"Average idle time: {avg_idle_time:.2f} ms")
    print(f"Average collisions: {avg_collision_count:.2f}")

if __name__ == "__main__":
    try:
        num_runs = 10 
        run_test_cases(num_runs)
    except Exception as e:
        print("An error occurred:", e)
        pygame.quit()
        sys.exit()
