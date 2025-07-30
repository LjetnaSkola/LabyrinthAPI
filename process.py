import cv2
import numpy as np
import click

def process_image(image_name):
    # ucitavanje i pretvaranje u binarnu crno-bijelu sliku
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Parametri lavirinta (mozda trebaju negdje drugo
    rows, cols = 10, 8
    wall_size = 3 * 4
    block_h = (binary.shape[0]- wall_size)/ rows
    block_w = (binary.shape[1]- wall_size)/ cols

    # init matrice pravaca
    directions_matrix = [[set() for _ in range(cols)] for _ in range(rows)]

    # detekcija zidova
    for i in range(rows):
        for j in range(cols):
            top = round(i * (block_h+ 0))
            left = round(j * (block_w+ 0))


            # GORE?
            wall_region = binary[top:top + wall_size, left :left + round(block_w)]
            if np.mean(wall_region) > 180:
                directions_matrix[i][j].add('U')

            # DOLE?
            wall_region = binary[top + round(block_h):top + round(block_h) + wall_size, left:left + round(block_w)]
            if np.mean(wall_region) > 180:
                directions_matrix[i][j].add('D')

            # LIJEVO?
            wall_region = binary[top:top + round(block_h), left:left + wall_size]
            if np.mean(wall_region) > 180:
                directions_matrix[i][j].add('L')

            # DESNO?
            wall_region = binary[top:top + round(block_h) , left+ round(block_w):left + round(block_w) + wall_size]
            if np.mean(wall_region) > 180:
                directions_matrix[i][j].add('R')
    return directions_matrix

def build_adjacency_list(maze):
    rows = len(maze)
    cols = len(maze[0])
    adj = {}

    # Pravci kao koordinatne razlike
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }

    for i in range(rows):
        for j in range(cols):
            neighbors = []
            for d in maze[i][j]:
                di, dj = directions[d]
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append((ni, nj))
            adj[(i, j)] = neighbors

    return adj

def wall_follower_directions(adjacency, start, goal):
    directions = ['R', 'D', 'L', 'U']
    delta = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    reverse_delta = {v: k for k, v in delta.items()}

    def get_direction(from_cell, to_cell):
        di = to_cell[0] - from_cell[0]
        dj = to_cell[1] - from_cell[1]
        return reverse_delta.get((di, dj))

    current = start
    if not adjacency[current]:
        return []

    direction = get_direction(current, adjacency[current][0])
    if direction is None:
        return []

    path = [current]
    directions_path = []

    while current != goal:
        dir_index = directions.index(direction)
        priority = [
            directions[(dir_index + 1) % 4],   # desno
            directions[dir_index],            # pravo
            directions[(dir_index - 1) % 4],  # lijevo
            directions[(dir_index + 2) % 4],  # nazad
        ]

        moved = False
        for d in priority:
            dx, dy = delta[d]
            ni, nj = current[0] + dx, current[1] + dy
            neighbor = (ni, nj)
            if neighbor in adjacency[current]:
                if len(path) >= 2 and neighbor == path[-2]:
                    path.pop()  # враћање назад
                    directions_path.pop()
                else:
                    path.append(neighbor)
                    directions_path.append(d)
                current = neighbor
                direction = d
                moved = True
                break

        if not moved:
            return []  # Zaglavljen

    return directions_path

def aggregate_directions(direction_list):
    if not direction_list:
        return []

    aggregated = []
    current_dir = direction_list[0]
    count = 1

    for d in direction_list[1:]:
        if d == current_dir:
            count += 1
        else:
            aggregated.append((count, current_dir))
            current_dir = d
            count = 1

    aggregated.append((count, current_dir))  # додај задњу групу

    return aggregated

def wall_follower_clean(adjacency, start, goal, follow_left=True):
    directions = ['R', 'D', 'L', 'U']
    delta = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    def turn_right(d): return directions[(directions.index(d) + 1) % 4]
    def turn_left(d): return directions[(directions.index(d) - 1) % 4]
    def turn_back(d): return directions[(directions.index(d) + 2) % 4]

    def move(pos, direction):
        dx, dy = delta[direction]
        return (pos[0] + dx, pos[1] + dy)

    current = start
    current_dir = 'R'
    path = []

    visited_states = []
    step = 0
    max_steps = 10000

    while current != goal and step < max_steps:
        state = (current, current_dir)
        if state in visited_states:
            break  # Infinite loop detected
        visited_states.append(state)

        step += 1

        if follow_left:
            check_dirs = [turn_left(current_dir), current_dir, turn_right(current_dir), turn_back(current_dir)]
        else:
            check_dirs = [turn_right(current_dir), current_dir, turn_left(current_dir), turn_back(current_dir)]

        moved = False
        moved_back = False
        for d in check_dirs:
            next_pos = move(current, d)
            if next_pos in adjacency.get(current, []):
                current_dir = d
                current = next_pos
                if not path or turn_back(d) != path[-1]:
                    path.append(d)
                else:
                    moved_back = True
                    path.pop(-1)
                moved = True
                break

        if not moved:
            break  # No possible moves

    return path, current, step



def solve_maze(image_name):
    maze = process_image(image_name)

    adj = build_adjacency_list(maze)
    start = (0, 0)
    goal = (9, 7)

    path = wall_follower_clean(adj, start, goal)
    return aggregate_directions(path[0])

@click.command()
@click.argument('image_name', required=True )
def main(image_name):
    path = solve_maze(image_name)
    print(f"Path for {image_name}:", path)

if __name__ == "__main__":
    main()