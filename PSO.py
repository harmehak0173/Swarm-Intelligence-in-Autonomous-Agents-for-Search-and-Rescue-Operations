# Harmehak Singh Khangura
import pygame
import numpy as np
import sys
import math
from collections import deque

pygame.init()

Screen_width = 800
Screen_height = 600
Grid_size = 40

Cells_X = Screen_width // Grid_size
Cells_Y = Screen_height // Grid_size

Agent_size = 10
Target_size = 15
Agents = 20
Targets = 5
Obstacles = 50

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
Max_speed = 5

obstacle_grid = np.zeros((Cells_X, Cells_Y), dtype=bool)

for _ in range(Obstacles):
    ox, oy = np.random.randint(0, Cells_X), np.random.randint(0, Cells_Y)
    obstacle_grid[ox, oy] = True

start_time = 0
total_time = 0
distance_travelled = 0
exploration_coverage = 0
pheromone_utilization = 0
idle_time = 0
collision_count = 0

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
            if neighbor.best_distance < local_best_distance and neighbor != self and not neighbor.has_found_target:
                local_best_distance = neighbor.best_distance
                local_best_x = neighbor.best_x
                local_best_y = neighbor.best_y

        r1 = np.random.rand()
        r2 = np.random.rand()

        cognitive_x = C1 * r1 * (self.best_x - self.x)
        social_x = C2 * r2 * (local_best_x - self.x)
        self.vx = W * self.vx + cognitive_x + social_x + np.random.uniform(-1, 1)

        cognitive_y = C1 * r1 * (self.best_y - self.y)
        social_y = C2 * r2 * (local_best_y - self.y)
        self.vy = W * self.vy + cognitive_y + social_y + np.random.uniform(-1, 1)

        speed = math.hypot(self.vx, self.vy)
        if speed > Max_speed:
            self.vx = (self.vx / speed) * Max_speed
            self.vy = (self.vy / speed) * Max_speed

    def avoid_obstacles(self):
        future_x = self.x + self.vx * 2
        future_y = self.y + self.vy * 2

        grid_x = int(future_x // Grid_size)
        grid_y = int(future_y // Grid_size)

        if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
            if obstacle_grid[grid_x, grid_y]:
                angle = math.atan2(self.vy, self.vx)
                avoid_angle = angle + math.pi / 2
                avoid_vx = math.cos(avoid_angle) * Max_speed
                avoid_vy = math.sin(avoid_angle) * Max_speed

                self.vx += avoid_vx
                self.vy += avoid_vy

                speed = math.hypot(self.vx, self.vy)
                if speed > Max_speed:
                    self.vx = (self.vx / speed) * Max_speed
                    self.vy = (self.vy / speed) * Max_speed

    def move(self, agents):
        current_time = pygame.time.get_ticks()
        self.avoid_obstacles()

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        grid_x = int(new_x // Grid_size)
        grid_y = int(new_y // Grid_size)

        if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
            if not obstacle_grid[grid_x, grid_y]:
                collision = False
                for agent in agents:
                    if agent != self:
                        if math.hypot(agent.x - new_x, agent.y - new_y) < Agent_size * 2:
                            global collision_count
                            collision_count += 1
                            collision = True
                            break
                if not collision:
                    self.x = new_x
                    self.y = new_y
                    self.distance_traveled += math.hypot(self.vx, self.vy)
                    self.is_idle = False
                else:
                    if not self.is_idle:
                        self.is_idle = True
                        self.last_state_change_time = current_time
                    else:
                        self.idle_time += current_time - self.last_state_change_time
                        self.last_state_change_time = current_time
            else:
                self.vx = -self.vx * 0.5 + np.random.uniform(-1, 1)
                self.vy = -self.vy * 0.5 + np.random.uniform(-1, 1)
                self.is_idle = False
        else:
            self.vx = -self.vx * 0.5 + np.random.uniform(-1, 1)
            self.vy = -self.vy * 0.5 + np.random.uniform(-1, 1)

        self.x += self.vx
        self.y += self.vy
        self.distance_traveled += math.hypot(self.vx, self.vy)

        self.x = max(0, min(self.x, Screen_width - 1))
        self.y = max(0, min(self.y, Screen_height - 1))

        grid_x = int(self.x // Grid_size)
        grid_y = int(self.y // Grid_size)
        if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
            explored_cells[grid_x, grid_y] = 1

    def check_for_target(self):
        found_new_target = False
        for target in self.targets[:]:
            distance = math.hypot(self.x - target.x, self.y - target.y)
            if distance < Target_size:
                self.targets.remove(target)
                self.found_targets.append(target)
                print("Target found!")
                self.has_found_target = True  
                self.best_distance = float('inf')  
                found_new_target = True
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
        color = Green if self.has_found_target else Blue  
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), Agent_size)

class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, Red, (int(self.x), int(self.y)), Target_size)

def draw_grid(screen):
    for x in range(0, Screen_width, Grid_size):
        pygame.draw.line(screen, Light_gray, (x, 0), (x, Screen_height))
    for y in range(0, Screen_height, Grid_size):
        pygame.draw.line(screen, Light_gray, (0, y), (Screen_width, y))

def draw_obstacles(screen):
    for i in range(Cells_X):
        for j in range(Cells_Y):
            if obstacle_grid[i, j]:
                pygame.draw.rect(
                    screen,
                    Dark_gray,
                    (i * Grid_size, j * Grid_size, Grid_size, Grid_size)
                )

def run_simulation():
    global start_time, total_time, distance_travelled, exploration_coverage, idle_time, collision_count

    screen = pygame.display.set_mode((Screen_width, Screen_height))
    pygame.display.set_caption("Search and Rescue Simulation")

    font = pygame.font.SysFont(None, 24)

    agents = []
    targets = []

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_r:
                    run_simulation()
                    return

        if not paused:
            screen.fill(White)
            draw_grid(screen)
            draw_obstacles(screen)

            for agent in agents:
                agent.update_neighbors(agents)

            for agent in agents:
                agent.update_velocity()
                agent.move(agents)
                agent.check_for_target()
                agent.draw(screen)

            for target in targets:
                target.draw(screen)

            if len(targets) == 0:
                total_time = pygame.time.get_ticks() - start_time
                distance_travelled = sum([agent.distance_traveled for agent in agents])

                total_idle_time = sum([agent.idle_time for agent in agents])
                idle_time = total_idle_time

                exploration_coverage = np.sum(explored_cells) / (Cells_X * Cells_Y) * 100

                pheromone_utilization = 0

                print(f"All targets found in {total_time} ms.")
                print(f"Total distance travelled: {distance_travelled}")
                print(f"Exploration coverage: {exploration_coverage}%")
                print(f"Pheromone utilization: {pheromone_utilization}%")
                print(f"Idle time: {idle_time} ms")
                print(f"Collisions: {collision_count}")
                running = False

            time_text = font.render(f"Time: {pygame.time.get_ticks() - start_time} ms", True, Black)
            screen.blit(time_text, (10, 10))
            target_text = font.render(f"Targets Remaining: {len(targets)}", True, Black)
            screen.blit(target_text, (10, 30))
            collision_text = font.render(f"Collisions: {collision_count}", True, Black)
            screen.blit(collision_text, (10, 50))

            instruction_text = font.render("Press 'P' to pause, 'R' to reset", True, Black)
            screen.blit(instruction_text, (10, Screen_height - 30))

            pygame.display.flip()
            clock.tick(15)
        else:
            pause_text = font.render("Paused. Press 'P' to resume.", True, Black)
            screen.blit(pause_text, (Screen_width // 2 - 100, Screen_height // 2))
            pygame.display.flip()
            clock.tick(15)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print("An error occurred:", e)
        pygame.quit()
        sys.exit()
