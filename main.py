from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Tuple
import shutil
import os
import cv2
import numpy as np
from process import solve_maze
from predict import predict_the_class

# --- CONFIG ---
SECRET_KEY = "tajni_kljuc"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- MOCK LOGIN ---
def authenticate_user(username: str, password: str):
    # Dummy provjera (увијек "user" / "password")
    return username == "user" and password == "password"

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- PLACEHOLDER: IMAGE PROCESSING ---
def process_image(image_path: str) -> List[List[str]]:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # labrnth params
    rows, cols = 10, 8
    wall_size = 3 * 4
    block_h = (binary.shape[0]- wall_size)/ rows # cell_size + wall_size
    block_w = (binary.shape[1]- wall_size)/ cols # cell_size + wall_size


    # cv2.imshow("Binary Image", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Иницијализација матрице праваца
    directions_matrix = [[set() for _ in range(cols)] for _ in range(rows)]

    # Детекција зидова
    for i in range(rows):
        for j in range(cols):
            top = round(i * (block_h+ 0))
            left = round(j * (block_w+ 0))


            # Провјера зида ГОРЕ

            wall_region = binary[top:top + wall_size, left :left + round(block_w)]
            if np.mean(wall_region) > 190:
                directions_matrix[i][j].add('U')

            # Провјера зида ДОЛЕ
            wall_region = binary[top + round(block_h):top + round(block_h) + wall_size, left:left + round(block_w)]
            if np.mean(wall_region) > 190:
                directions_matrix[i][j].add('D')

            # Провјера зида ЛИЈЕВО
            wall_region = binary[top:top + round(block_h), left:left + wall_size]
            if np.mean(wall_region) > 190:
                directions_matrix[i][j].add('L')

            # Провјера зида ДЕСНО
            wall_region = binary[top:top + round(block_h) , left+ round(block_w):left + round(block_w) + wall_size]
            if np.mean(wall_region) > 190:
                directions_matrix[i][j].add('R')
    return directions_matrix

# --- FROM PREVIOUS STEPS ---
def build_adjacency_list(matrix):
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    adj = {}

    for i in range(rows):
        for j in range(cols):
            cell = matrix[i][j]
            if not cell:
                continue
            neighbors = []
            for dir in cell:
                if dir in directions:
                    di, dj = directions[dir]
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and matrix[ni][nj]:
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
            directions[(dir_index + 1) % 4],
            directions[dir_index],
            directions[(dir_index - 1) % 4],
            directions[(dir_index + 2) % 4],
        ]

        moved = False
        for d in priority:
            dx, dy = delta[d]
            ni, nj = current[0] + dx, current[1] + dy
            neighbor = (ni, nj)
            if neighbor in adjacency[current]:
                if len(path) >= 2 and neighbor == path[-2]:
                    path.pop()
                    directions_path.pop()
                else:
                    path.append(neighbor)
                    directions_path.append(d)
                current = neighbor
                direction = d
                moved = True
                break
        if not moved:
            return []
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
    aggregated.append((count, current_dir))
    return aggregated

# --- LOGIN ENDPOINT ---
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- SOLVE ENDPOINT ---
@app.post("/solve")
async def solve_maze_endpoint(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    filename = f"temp_images/{file.filename}"
    os.makedirs("temp_images", exist_ok=True)
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    path = []
    try:
        # image_path = rotate_and_save_image(image_path) # pošto je kamera pogrešno zalijepljena :D
        predicted_class = predict_the_class(filename)
        # ovdje može da ide ručni posao:
        #  - ukoliko je slika dobro predviđena - ubaciti je u odgovarajući train folder
        #  - prije neke sljedeće predikcije - istrenirati!
        #  - a i ne mora
        image_name = os.path.join("original_lavirint", f"slika{predicted_class + 1}.png")
        path = solve_maze(image_name)
        print(f"Path for {image_name}:", path)

    finally:
        os.remove(filename)

    return {"path": path}