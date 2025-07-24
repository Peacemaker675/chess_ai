import chess
import stockfish
import csv
import random
import pandas as pd

STOCKFISH_PATH = ""
OUTPUT_FILE = "dataset_final.csv"
NUMBER_OF_FEN = 10000
MAX_MOVES_PER_GAME = 50

fish = stockfish.Stockfish(STOCKFISH_PATH)

def generate_random_board():
    board = chess.Board()
    num_of_moves = random.randint(1, MAX_MOVES_PER_GAME)
    for _ in range(num_of_moves):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        board.push(move)
    return board

def main():
    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["FEN", "score_type", "score"])
        for _ in range(NUMBER_OF_FEN):
            try:
                board = generate_random_board()
                fen = board.fen()
                fish.set_fen_position(fen)
                evaluation = fish.get_evaluation()
                writer.writerow([fen, evaluation["type"], evaluation["value"]])
            except Exception as e:
                print(f"Error processing FEN {fen}: {e}")

if __name__ == "__main__":
    main()
    df = pd.read_csv(OUTPUT_FILE)
    df.drop_duplicates(subset=["FEN"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
