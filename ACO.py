#Harmehak Singh Khangura
import pygame
import numpy as np
from collections import deque
import sys
import math
import colorsys  

pygame.init()


Screen_width = 800
Screen_length = 600
Grid_size = 40  

Cells_X = Screen_width // Grid_size  
Cells_Y = Screen_length // Grid_size  

Agent_size = 10
Target_size = 15
Agents = 20
Targets = 5
Obstacles = 50  

White = (255, 255, 255)
Light_gray = (200, 200, 200)
Gray = (100, 100, 100)
Dark_gray = (80, 80, 80)  
Black = (0, 0, 0)
Blue = (0, 0, 255)
Green = (0, 255, 0)  
Red = (255, 0, 0)


Max_pheromone_strength = 255  
Pheromone_increment = 50   
Decay_rate = 0.005         
Low_Threshold = 0.5 


Stuck_threshold = 30    
Explore_timeout = 3000  


pheromone_grid = np.zeros((Cells_X, Cells_Y))


obstacle_grid = np.zeros((Cells_X, Cells_Y), dtype=bool)


for i in range(Obstacles):
    ox, oy = np.random.randint(0, Cells_X), np.random.randint(0, Cells_Y)
    obstacle_grid[ox, oy] = True


start_time = 0
total_time = 0
distance_travelled = 0
target_discovery_efficiency = 0
pheromone_utilization = 0
exploration_coverage = 0
idle_time = 0
collision_count = 0


explored_cells = np.zeros((Cells_X, Cells_Y))

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 1.5  
        self.found_target = False  
        self.distance_traveled = 0  
        self.visited_cells = deque(maxlen=50)  
        self.last_exploration_time = pygame.time.get_ticks()
        self.following_pheromone = False
        self.is_following_pheromone = False  
        self.is_idle = False  
        self.time_following_pheromone = 0  
        self.time_idle = 0  
        self.last_state_change_time = pygame.time.get_ticks()

    def is_stuck(self):
        if len(self.visited_cells) < self.visited_cells.maxlen:
            return False
        most_common_cell = max(set(self.visited_cells), key=self.visited_cells.count)
        count = self.visited_cells.count(most_common_cell)
        return count >= Stuck_threshold

    def deposit_pheromone(self):
        global pheromone_grid  
        x_clamped = max(0, min(self.x, Screen_width - 1))
        y_clamped = max(0, min(self.y, Screen_length - 1))
        grid_x = int(x_clamped // Grid_size)
        grid_y = int(y_clamped // Grid_size)
        if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y:
            pheromone_grid[grid_x, grid_y] = min(
                pheromone_grid[grid_x, grid_y] + Pheromone_increment, Max_pheromone_strength
            )

    def get_strongest_pheromone(self):
        global pheromone_grid 
        x_clamped = max(0, min(self.x, Screen_width - 1))
        y_clamped = max(0, min(self.y, Screen_length - 1))
        grid_x = int(x_clamped // Grid_size)
        grid_y = int(y_clamped // Grid_size)

        max_strength = 0
        best_position = None

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < Cells_X and 0 <= ny < Cells_Y:
                    if pheromone_grid[nx, ny] > max_strength and not obstacle_grid[nx, ny]:
                        max_strength = pheromone_grid[nx, ny]
                        best_position = (nx * Grid_size + Grid_size // 2, ny * Grid_size + Grid_size // 2)
        return best_position

    def avoid_obstacles(self, desired_dx, desired_dy):
        angle = math.atan2(desired_dy, desired_dx)
        for i in range(-3, 4):  
            new_angle = angle + (i * math.pi / 16)
            dx = math.cos(new_angle) * self.speed
            dy = math.sin(new_angle) * self.speed
            new_x = self.x + dx
            new_y = self.y + dy
            new_x = max(0, min(new_x, Screen_width - 1))
            new_y = max(0, min(new_y, Screen_length - 1))
            grid_x = int(new_x // Grid_size)
            grid_y = int(new_y // Grid_size)
            if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y and not obstacle_grid[grid_x, grid_y]:
                return new_x, new_y  
        return self.x, self.y  

    def move(self, targets, other_agents):
        global idle_time, collision_count

        current_time = pygame.time.get_ticks()

        
        self.x = max(0, min(self.x, Screen_width - 1))
        self.y = max(0, min(self.y, Screen_length - 1))

        if self.found_target:
            
            if not self.is_idle:
                self.is_idle = True
                self.last_state_change_time = current_time
            else:
                self.time_idle += current_time - self.last_state_change_time
                self.last_state_change_time = current_time
            return

        grid_x = int(self.x // Grid_size)
        grid_y = int(self.y // Grid_size)

        self.visited_cells.append((grid_x, grid_y))
        explored_cells[grid_x, grid_y] = 1

        if self.is_stuck():
            if not self.is_idle:
                self.is_idle = True
                self.last_state_change_time = current_time
            else:
                self.time_idle += current_time - self.last_state_change_time
                self.last_state_change_time = current_time

            self.random_move()
            self.last_exploration_time = current_time
            self.is_following_pheromone = False
            self.following_pheromone = False
            return

        
        for agent in other_agents:
            if agent != self:
                if math.hypot(agent.x - self.x, agent.y - self.y) < Agent_size * 2:
                    collision_count += 1

        
        for target in targets:
            distance_to_target = math.hypot(self.x - target.x, self.y - target.y)
            if distance_to_target < 200:
                if self.is_idle:
                    self.is_idle = False
                    self.last_state_change_time = current_time

                if self.is_following_pheromone:
                    self.is_following_pheromone = False
                    self.last_state_change_time = current_time

                self.move_toward(target.x, target.y)
                self.deposit_pheromone()
                self.last_exploration_time = current_time
                self.following_pheromone = False
                return

        if current_time - self.last_exploration_time > Explore_timeout:
            if not self.is_idle:
                self.is_idle = True
                self.last_state_change_time = current_time
            else:
                self.time_idle += current_time - self.last_state_change_time
                self.last_state_change_time = current_time

            self.random_move()
            self.last_exploration_time = current_time
            self.is_following_pheromone = False
            self.following_pheromone = False
            return

        
        best_position = self.get_strongest_pheromone()
        if best_position:
            if self.is_idle:
                self.is_idle = False
                self.last_state_change_time = current_time

            if not self.is_following_pheromone:
                self.is_following_pheromone = True
                self.last_state_change_time = current_time
            else:
                self.time_following_pheromone += current_time - self.last_state_change_time
                self.last_state_change_time = current_time

            self.move_toward(*best_position)
            self.deposit_pheromone()
            self.following_pheromone = True
        else:
            if not self.is_idle:
                self.is_idle = True
                self.last_state_change_time = current_time
            else:
                self.time_idle += current_time - self.last_state_change_time
                self.last_state_change_time = current_time

            if self.is_following_pheromone:
                self.is_following_pheromone = False
                self.last_state_change_time = current_time

            self.random_move()
            self.following_pheromone = False

        
        self.x = max(0, min(self.x, Screen_width - 1))
        self.y = max(0, min(self.y, Screen_length - 1))

    def random_move(self):
        for _ in range(8):  
            angle = np.random.uniform(0, 2 * np.pi)
            dx = math.cos(angle) * self.speed
            dy = math.sin(angle) * self.speed
            new_x = self.x + dx
            new_y = self.y + dy
           
            new_x = max(0, min(new_x, Screen_width - 1))
            new_y = max(0, min(new_y, Screen_length - 1))
            grid_x = int(new_x // Grid_size)
            grid_y = int(new_y // Grid_size)
            if 0 <= grid_x < Cells_X and 0 <= grid_y < Cells_Y and not obstacle_grid[grid_x, grid_y]:
                self.x = new_x
                self.y = new_y
                self.deposit_pheromone()
                self.distance_traveled += self.speed
                return
        pass

    def move_toward(self, target_x, target_y):
        direction_x = target_x - self.x
        direction_y = target_y - self.y
        distance = math.hypot(direction_x, direction_y)
        if distance > 0:
            desired_dx = (direction_x / distance) * self.speed
            desired_dy = (direction_y / distance) * self.speed
            new_x, new_y = self.avoid_obstacles(desired_dx, desired_dy)
            if (new_x, new_y) != (self.x, self.y):
                self.x = new_x
                self.y = new_y
                self.distance_traveled += self.speed
            else:
                pass

    def check_for_target(self, targets):
        if self.found_target:
            return

        for target in targets[:]: 
            distance = math.hypot(self.x - target.x, self.y - target.y)
            if distance < Target_size:
                targets.remove(target)
                self.found_target = True
                print("Target found!")
                break

    def draw(self, screen):
        color = Green if self.found_target else Blue
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), Agent_size)

class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, Red, (int(self.x), int(self.y)), Target_size)

def draw_grid(screen):
    for x in range(0, Screen_width, Grid_size):
        pygame.draw.line(screen, Light_gray, (x, 0), (x, Screen_length))
    for y in range(0, Screen_length, Grid_size):
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

def decay_pheromones():
    global pheromone_grid  
    pheromone_grid *= (1 - Decay_rate)
    pheromone_grid[pheromone_grid < Low_Threshold] = 0  

def pheromone_color(strength):
    hue = 0.8 * (1 - strength / Max_pheromone_strength)  
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)               # Used from https://stackoverflow.com/questions/40948069/color-range-python
    return int(r * 255), int(g * 255), int(b * 255)

def run_simulation():
    global start_time, total_time, distance_travelled, target_discovery_efficiency, pheromone_utilization, exploration_coverage, idle_time, collision_count

    
    screen = pygame.display.set_mode((Screen_width, Screen_length))
    pygame.display.set_caption("Swarm Intelligence - Search and Rescue")

    
    font = pygame.font.SysFont(None, 24)

    
    agents = []
    targets = []

    
    while len(agents) < Agents:
        x = np.random.uniform(0, Screen_width - 1)
        y = np.random.uniform(0, Screen_length - 1)
        grid_x = int(x // Grid_size)
        grid_y = int(y // Grid_size)
        if not obstacle_grid[grid_x, grid_y]:
            agents.append(Agent(x, y))

    
    while len(targets) < Targets:
        x = np.random.uniform(0, Screen_width - 1)
        y = np.random.uniform(0, Screen_length - 1)
        grid_x = int(x // Grid_size)
        grid_y = int(y // Grid_size)
        if not obstacle_grid[grid_x, grid_y]:
            targets.append(Target(x, y))

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
            
            decay_pheromones()

            
            screen.fill(White)

            
            draw_grid(screen)

            
            draw_obstacles(screen)

            
            for i in range(Cells_X):
                for j in range(Cells_Y):
                    if pheromone_grid[i, j] > 0:
                        color = pheromone_color(pheromone_grid[i, j])
                        pygame.draw.rect(
                            screen,
                            color,
                            pygame.Rect(i * Grid_size, j * Grid_size, Grid_size, Grid_size)
                        )

            
            for target in targets:
                target.draw(screen)

            
            for agent in agents:
                agent.move(targets, agents)
                agent.check_for_target(targets)
                agent.draw(screen)

            if len(targets) == 0:
                total_time = pygame.time.get_ticks() - start_time
                distance_travelled = sum([agent.distance_traveled for agent in agents])

                total_following_time = sum([agent.time_following_pheromone for agent in agents])

                pheromone_utilization = (total_following_time / (total_time * Agents)) * 100

                
                total_idle_time = sum([agent.time_idle for agent in agents])
                idle_time = total_idle_time  

                exploration_coverage = np.sum(explored_cells) / (Cells_X * Cells_Y) * 100
                target_discovery_efficiency = Targets / total_time * 1000  

                print(f"All targets found in {total_time} ms.")
                print(f"Total distance travelled: {distance_travelled} pixels")
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
            screen.blit(instruction_text, (10, Screen_length - 30))

            
            pygame.display.flip()
            clock.tick(60)
        else:
            pause_text = font.render("Paused. Press 'P' to resume.", True, Black)
            screen.blit(pause_text, (Screen_width // 2 - 100, Screen_length // 2))
            pygame.display.flip()
            clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"An error occurRed")
        pygame.quit()
        sys.exit()