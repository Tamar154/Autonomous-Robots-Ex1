import pygame
import numpy as np
import heapq
from collections import defaultdict
import random
import math

class Drone:
    def __init__(self, start_position, map_data, far_point):
        self.position = start_position
        self.start_position = start_position
        self.map_data = map_data
        self.far_point = far_point
        self.battery_level = 100  # Start with full battery
        self.covered_area = set()
        self.path = [start_position]
        self.time_elapsed = 0
        self.current_direction = (1, 0)  # Start moving right
        self.returning_home = False
        self.return_path = []
        self.adjacency_list = defaultdict(list)
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.velocity = (0, 0)  # (vx, vy)
        self.altitude = 0
        self.flight_state = "Taking off"

    def get_sensor_data(self):
        """Get sensor data for the drone."""
        # Define the sensor directions relative to the drone's orientation
        directions = {
            'd0': (0, -1),  # front
            'd1': (0, 1),   # back
            'd2': (-1, 0),  # left
            'd3': (1, 0),   # right
            # 'd4': (1, 1),   # front-right
            # 'd5': (-1, -1)  # back-left
        }

        # Rotate the sensor directions based on the drone's yaw
        rotated_directions = {key: self._rotate_vector(vec, self.yaw) for key, vec in directions.items()}

        # Calculate distances to obstacles
        distances = {key: self._distance_to_obstacle(self.position, direction) for key, direction in rotated_directions.items()}

        sensor_data = {
            'd0': distances['d0'],
            'd1': distances['d1'],
            'd2': distances['d2'],
            'd3': distances['d3'],
            # 'd4': distances['d4'],
            # 'd5': distances['d5'],
            'yaw': self.yaw,
            'Vx': self.velocity[0] * 0.025,  # Convert to m/s
            'Vy': self.velocity[1] * 0.025,  # Convert to m/s
            'Z': self.altitude,
            'baro': self.altitude,
            'bat': self.battery_level,
            'pitch': self.pitch,
            'roll': self.roll,
            'accX': self.velocity[0] * 10 * 0.025,  # Simulate acceleration data in m/s^2
            'accY': self.velocity[1] * 10 * 0.025,  # Simulate acceleration data in m/s^2
            'accZ': 0  # Placeholder for vertical acceleration
        }
        return sensor_data

    def _distance_to_obstacle(self, position, direction):
        x, y = position
        dx, dy = direction
        distance = 0
        max_detection_range = 4  # 4 pixels corresponding to 10 cm
        while distance < max_detection_range:
            x += dx
            y += dy
            x_int = int(round(x))
            y_int = int(round(y))
            if not (0 <= x_int < self.map_data.shape[1] and 0 <= y_int < self.map_data.shape[0]):
                break
            if self.map_data[y_int, x_int] == 0:
                break
            distance += math.sqrt(dx**2 + dy**2)
        # Adding sensor error
        distance += distance * random.uniform(-0.02, 0.02)  # +-2% error
        return min(distance, max_detection_range)



    def move(self, direction):
        """Move the drone in the given direction."""
        if direction is None:
            return

        x, y = self.position
        dx, dy = direction
        new_position = (x + dx, y + dy)
        if self._is_valid_position(new_position):
            self.position = new_position
            self.covered_area.add(new_position)
            self.path.append(new_position)
            self._update_adjacency_list((x, y), new_position)
            self.update_orientation(direction)
            self.update_velocity(direction)
            print(f"Moved to new position: {self.position}")
        else:
            print(f"Failed to move to new position: {new_position}, trying different directions")
            possible_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for d in possible_directions:
                new_position = (x + d[0], y + d[1])
                if self._is_valid_position(new_position):
                    self.position = new_position
                    self.covered_area.add(new_position)
                    self.path.append(new_position)
                    self._update_adjacency_list((x, y), new_position)
                    self.current_direction = d
                    self.update_orientation(d)
                    self.update_velocity(d)
                    print(f"Moved to new position: {self.position} after detecting obstacle")
                    break

    def _update_adjacency_list(self, old_position, new_position):
        """Update the adjacency list for the drone's path."""
        self.adjacency_list[old_position].append(new_position)
        self.adjacency_list[new_position].append(old_position)

    def _is_valid_position(self, position):
        """Check if a position is valid (i.e., within bounds and not an obstacle)."""
        x, y = position
        valid = 0 <= x < self.map_data.shape[1] and 0 <= y < self.map_data.shape[0] and self.map_data[y, x] == 1
        print(f"Position {position} is {'valid' if valid else 'invalid'}")
        return valid

    def update_battery(self, time_step=1):
        """Update the battery level based on time step."""
        self.battery_level -= (100 / 480) * time_step
        print(f"Battery updated: {self.battery_level:.2f}%")
        if self.battery_level <= 50 and not self.returning_home:
            print("Battery at 50%, stopping drone.")
            self.flight_state = "Stopped"

    def plan_next_move(self):
        """Plan the next move based on the current state and battery level."""
        x, y = self.position
        print(f"Current position: {self.position}, Battery: {self.battery_level:.2f}%")

        # Spiral movement pattern
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # down, right, up, left
        for direction in directions:
            next_position = (x + direction[0], y + direction[1])
            if self._is_valid_position(next_position) and next_position not in self.covered_area:
                print(f"Next move: {direction}")
                return direction

        # If no unvisited adjacent cells in spiral pattern, fall back to random unvisited adjacent move
        unvisited_adjacent = []
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_position = (x + direction[0], y + direction[1])
            if self._is_valid_position(next_position) and next_position not in self.covered_area:
                unvisited_adjacent.append(direction)

        if unvisited_adjacent:
            next_move = random.choice(unvisited_adjacent)
            print(f"Unvisited adjacent: {unvisited_adjacent}, Next move: {next_move}")
            return next_move

        print(f"Continuing in current direction: {self.current_direction}")
        return self.current_direction


    def start_returning_home(self):
        """Start returning home using Dijkstra's algorithm."""
        self.returning_home = True
        self.return_path = self.dijkstra(self.position, self.start_position)
        print("Starting to return home, path:", self.return_path)

    def return_home(self):
        """Return home following the calculated path."""
        if not self.return_path:
            print("Reached home or no valid path")
            return None

        next_position = self.return_path.pop(0)
        move = (next_position[0] - self.position[0], next_position[1] - self.position[1])
        print(f"Returning home, next move: {move}")
        return move

    def dijkstra(self, start, goal):
        """Dijkstra's algorithm to find the shortest path to the goal."""
        queue = [(0, start)]
        distances = {start: 0}
        previous_nodes = {start: None}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == goal:
                break

            for neighbor in self.adjacency_list[current_node]:
                distance = current_distance + 1
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    priority = distance
                    heapq.heappush(queue, (priority, neighbor))
                    previous_nodes[neighbor] = current_node

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous_nodes[current]

        path.reverse()
        print(f"Path found: {path}")
        return path if path[0] == start else []

    def update_orientation(self, direction):
        """Update the drone's orientation based on its direction."""
        dx, dy = direction
        if dx > 0:
            self.yaw = 0
        elif dx < 0:
            self.yaw = 180
        elif dy > 0:
            self.yaw = 90
        elif dy < 0:
            self.yaw = -90

        self.pitch = math.atan2(dy, 1) * (180 / math.pi)  # Simplified pitch calculation
        self.roll = math.atan2(dx, 1) * (180 / math.pi)   # Simplified roll calculation

    def update_velocity(self, direction):
        """Update the drone's velocity based on its direction."""
        vx, vy = direction
        self.velocity = (vx * 3, vy * 3)  # Assume maximum speed is 3 m/s for each direction component

    def _rotate_vector(self, vector, angle_degrees):
        """Rotate a 2D vector by a given angle in degrees."""
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        x, y = vector
        x_new = x * cos_angle - y * sin_angle
        y_new = x * sin_angle + y * cos_angle
        return (x_new, y_new)

def run_simulation(drone, window_size=(800, 600), scale_factor=1):
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption('Drone Simulation')

    clock = pygame.time.Clock()

    # Load drone image
    drone_image = pygame.image.load('drone.png')
    drone_image = pygame.transform.scale(drone_image, (20, 20))

    # Calculate scaling factors based on the map and window size
    map_height, map_width = drone.map_data.shape
    scale_x = window_size[0] / map_width
    scale_y = window_size[1] / map_height
    scale = min(scale_x, scale_y)

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (200, 200, 200)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)

    sensor_colors = {
        'd0': RED,
        'd1': GREEN,
        'd2': BLUE,
        'd3': YELLOW,
        # 'd4': ORANGE,
        # 'd5': CYAN
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # Draw map
        for y in range(map_height):
            for x in range(map_width):
                color = WHITE if drone.map_data[y, x] == 1 else BLACK
                rect = pygame.Rect(x * scale, y * scale, scale, scale)
                pygame.draw.rect(screen, color, rect)

        # Draw covered area
        for pos in drone.covered_area:
            x, y = pos
            rect = pygame.Rect(x * scale, y * scale, scale, scale)
            pygame.draw.rect(screen, GRAY, rect)

        # Draw path
        for pos in drone.path:
            x, y = pos
            rect = pygame.Rect(x * scale, y * scale, scale, scale)
            pygame.draw.rect(screen, GREEN, rect)

        # Draw drone
        drone_x, drone_y = drone.position
        drone_rect = drone_image.get_rect(center=((drone_x + 0.5) * scale, (drone_y + 0.5) * scale))
        screen.blit(drone_image, drone_rect)

        # Get sensor data
        sensor_data = drone.get_sensor_data()

        # Draw distance sensors
        directions = {
            'd0': (0, -1),  # up
            'd1': (0, 1),   # down
            'd2': (-1, 0),  # left
            'd3': (1, 0),   # right
            # 'd4': (1, 1),   # forward
            # 'd5': (-1, -1)  # backward
        }
        for direction_name, distance in sensor_data.items():
            if direction_name in directions:
                direction = drone._rotate_vector(directions[direction_name], drone.yaw)
                start_pos = ((drone_x + 0.5) * scale, (drone_y + 0.5) * scale)
                max_sensor_fraction = 0.5  # Fraction of the actual detected distance to draw the sensor line
                end_pos = (drone_x + direction[0] * distance * max_sensor_fraction, drone_y + direction[1] * distance * max_sensor_fraction)
                end_pos = ((end_pos[0] + 0.5) * scale, (end_pos[1] + 0.5) * scale)  # Scale end position
                color = sensor_colors[direction_name]
                pygame.draw.line(screen, color, start_pos, end_pos, 2)

        pygame.display.flip()
        clock.tick(5) 

        # Simulation logic
        if drone.flight_state == "Stopped":
            print("Drone has stopped due to low battery.")
            running = False
            continue

        drone.update_battery()
        if drone.battery_level <= 50:
            drone.flight_state = "Stopped"
            continue

        next_move = drone.plan_next_move()
        drone.move(next_move)

    pygame.quit()


room_map = np.ones((30, 30), dtype=int)
room_map[0, :] = 0  # Top wall
room_map[-1, :] = 0  # Bottom wall
room_map[:, 0] = 0  # Left wall
room_map[:, -1] = 0  # Right wall
room_map[5:25, 10] = 0  # Vertical wall
room_map[5:25, 20] = 0  # Vertical wall
room_map[10, 5:10] = 0  # Horizontal wall
room_map[20, 20:25] = 0  # Horizontal wall
room_map[15, 10:20] = 0  # Horizontal wall in the middle
room_map[5, 15] = 0
room_map[10, 25] = 0
room_map[25, 5] = 0

start_position = (15, 14)
far_point = (29, 29)

drone = Drone(start_position, room_map, far_point)
run_simulation(drone)
