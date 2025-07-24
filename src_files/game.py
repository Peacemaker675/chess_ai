import os
import sys
import time
import pygame
import chess
import numpy as np
from tensorflow.keras.models import load_model

# Constants
WIDTH, HEIGHT = 600, 600
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
FPS = 30

# Colors
HIGHLIGHT_COLOR = (66, 135, 245, 100)
MOVE_COLOR = (80, 200, 120, 100)
LAST_MOVE_COLOR = (245, 66, 66, 100)

# ensure correct path for resources when bundled with PyInstaller
def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# Load the pre-trained model
model = load_model(resource_path("model/best_chess_model.h5"), compile=False)

# Map chess pieces to indices for tensor representation
piece_to_index = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_tensor(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 18), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = 7 - chess.square_rank(square), chess.square_file(square)
            tensor[row, col, piece_to_index[piece.symbol()]] = 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        row, col = 7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[row, col, 17] = 1.0
    return tensor

# Function to evaluate the board using the model
def evaluate_board(model, board: chess.Board) -> float:
    board_tensor = fen_to_tensor(board.fen())
    pred = model.predict(np.expand_dims(board_tensor, axis=0), verbose=0)
    return float(pred[0][0])

# Minimax algorithm with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximize, model):
    if depth == 0 or board.is_game_over():
        return evaluate_board(model, board), None
    best_move = None
    moves = list(board.legal_moves)
    np.random.shuffle(moves)
    if maximize:
        max_eval = -np.inf
        for m in moves:
            board.push(m)
            val, _ = minimax(board, depth-1, alpha, beta, False, model)
            board.pop()
            if val > max_eval:
                max_eval, best_move = val, m
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = np.inf
        for m in moves:
            board.push(m)
            val, _ = minimax(board, depth-1, alpha, beta, True, model)
            board.pop()
            if val < min_eval:
                min_eval, best_move = val, m
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move

# Function to load images for chess pieces
def load_images():
    images = {}
    pieces = ['wP','wR','wN','wB','wQ','wK','bP','bR','bN','bB','bQ','bK']
    for p in pieces:
        img = pygame.image.load(resource_path(f"images/{p}.png"))
        images[p] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))
    return images

# Function to convert chess square to screen coordinates
def square_to_screen(square, flip):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    if flip:
        screen_col = 7 - file
        screen_row = rank
    else:
        screen_col = file
        screen_row = 7 - rank
    
    return screen_row, screen_col

# Function to convert screen coordinates to chess square
def screen_to_square(screen_row, screen_col, flip):
    if flip:
        file = 7 - screen_col
        rank = screen_row
    else:
        file = screen_col
        rank = 7 - screen_row
    
    return chess.square(file, rank)

# Function to draw the chess board
def draw_board(screen, flip):
    colors = [pygame.Color('white'), pygame.Color('gray')]
    for screen_row in range(DIMENSION):
        for screen_col in range(DIMENSION):
            # Color pattern based on actual chess board squares
            square = screen_to_square(screen_row, screen_col, flip)
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            color_index = (file + rank) % 2
            
            pygame.draw.rect(screen, colors[color_index], 
                           pygame.Rect(screen_col * SQ_SIZE, screen_row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Function to draw pieces on the board
def draw_pieces(screen, board, images, flip):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            screen_row, screen_col = square_to_screen(square, flip)
            piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
            screen.blit(images[piece_key], (screen_col * SQ_SIZE, screen_row * SQ_SIZE))

# Function to draw highlights on the board
def draw_highlights(screen, squares, color, flip):
    s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
    s.fill(color)
    for square in squares:
        screen_row, screen_col = square_to_screen(square, flip)
        screen.blit(s, (screen_col * SQ_SIZE, screen_row * SQ_SIZE))

# Function to get the user's choice of color at the start
def get_start_choice(screen, clock):
    pygame.font.init()
    font = pygame.font.SysFont('segoeui', 36, True)
    title_font = pygame.font.SysFont('segoeui', 48, True)
    choices = {'White': chess.WHITE, 'Black': chess.BLACK}
    buttons = []
    for i, (text, col) in enumerate(choices.items()):
        btn = pygame.Rect(WIDTH//3 * (i+1) - 100, HEIGHT//2 - 40, 200, 80)
        buttons.append((btn, text, col))

    while True:
        screen.fill((30, 30, 40))
        for y in range(HEIGHT):
            alpha = int(255 * (y / HEIGHT))
            color = (50, 50, 80, alpha)
            surf = pygame.Surface((WIDTH, 1), pygame.SRCALPHA)
            surf.fill(color)
            screen.blit(surf, (0, y))

        shadow = title_font.render('Choose Your Color', True, (0, 0, 0))
        screen.blit(shadow, (WIDTH//2 - shadow.get_width()//2 + 2, HEIGHT//3 + 2))
        title = title_font.render('Choose Your Color', True, pygame.Color('white'))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3))

        mouse_pos = pygame.mouse.get_pos()
        for btn, text, col in buttons:
            hover = btn.collidepoint(mouse_pos)
            btn_color = (180, 180, 255) if hover else (200, 200, 255)
            pygame.draw.rect(screen, btn_color, btn, border_radius=12)
            pygame.draw.rect(screen, (100, 100, 150), btn, 4, border_radius=12)
            lbl = font.render(text, True, pygame.Color('black'))
            screen.blit(lbl, (btn.centerx - lbl.get_width()//2, btn.centery - lbl.get_height()//2))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for btn, text, col in buttons:
                    if btn.collidepoint(event.pos):
                        return col

        pygame.display.flip()
        clock.tick(FPS)

def main():
    pygame.init()
    pygame.mixer.init()
    move_sound = pygame.mixer.Sound(resource_path("audio/move.wav"))
    capture_sound = pygame.mixer.Sound(resource_path("audio/capture.wav"))
    move_sound.set_volume(1.0)
    capture_sound.set_volume(1.0)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    clock = pygame.time.Clock()
    images = load_images()

    user_color = get_start_choice(screen, clock)
    bot_color = not user_color
    flip = (user_color == chess.BLACK)

    board = chess.Board()
    depth = 1
    move_delay = 1.0
    last_move_time = time.time()
    last_move = None
    selected_square = None
    legal_move_squares = []

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and board.turn == user_color and not board.is_game_over():
                mx, my = e.pos
                screen_col = mx // SQ_SIZE
                screen_row = my // SQ_SIZE
                
                # Convert screen coordinates to chess square
                clicked_square = screen_to_square(screen_row, screen_col, flip)
                piece = board.piece_at(clicked_square)

                # If we have a selected square and clicked on a legal move destination
                if selected_square is not None and clicked_square in legal_move_squares:
                    move = chess.Move(selected_square, clicked_square)
                    # Handle pawn promotion (default to queen)
                    if move.promotion is None and board.piece_at(selected_square).piece_type == chess.PAWN:
                        if (board.piece_at(selected_square).color == chess.WHITE and chess.square_rank(clicked_square) == 7) or \
                           (board.piece_at(selected_square).color == chess.BLACK and chess.square_rank(clicked_square) == 0):
                            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)
                    
                    if move in board.legal_moves:
                        capture_sound.play() if board.is_capture(move) else move_sound.play()
                        board.push(move)
                        last_move = move
                        selected_square = None
                        legal_move_squares = []
                        last_move_time = time.time()
                
                # If clicked on own piece, select it
                elif piece and piece.color == user_color:
                    selected_square = clicked_square
                    legal_move_squares = [move.to_square for move in board.legal_moves 
                                        if move.from_square == clicked_square]
                
                # If clicked on empty square or opponent piece without selection, deselect
                else:
                    selected_square = None
                    legal_move_squares = []

        # Bot move
        if not board.is_game_over() and board.turn == bot_color and time.time() - last_move_time >= move_delay:
            _, move = minimax(board, depth, -np.inf, np.inf, board.turn == chess.WHITE, model)
            if move:
                capture_sound.play() if board.is_capture(move) else move_sound.play()
                board.push(move)
                last_move = move
                last_move_time = time.time()

        # Draw everything
        draw_board(screen, flip)
        
        # Highlight last move
        if last_move:
            draw_highlights(screen, [last_move.from_square, last_move.to_square], LAST_MOVE_COLOR, flip)
        
        # Highlight selected square and legal moves
        if selected_square is not None:
            draw_highlights(screen, [selected_square], HIGHLIGHT_COLOR, flip)
            draw_highlights(screen, legal_move_squares, MOVE_COLOR, flip)

        draw_pieces(screen, board, images, flip)
        pygame.display.flip()
        clock.tick(FPS)

        # Game over handling
        if board.is_game_over():
            result = board.result()
            if result == '1/2-1/2':
                result_text = 'Draw'
            elif result == '1-0':
                result_text = 'White wins'
            else:
                result_text = 'Black wins'
            
            font = pygame.font.SysFont('arial', 48, True)
            msg = font.render(result_text, True, pygame.Color('gold'))
            screen.blit(msg, (WIDTH//2 - msg.get_width()//2, HEIGHT//2 - msg.get_height()//2))
            pygame.display.flip()
            time.sleep(3)
            running = False

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()