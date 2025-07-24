import chess
import chess.pgn
import random
import csv
import numpy as np
import time
import concurrent.futures
from multiprocessing import cpu_count
import stockfish

def extract_positions_from_pgn(pgn_file_path, num_positions_per_phase):
    opening_positions = set()
    midgame_positions = set()
    endgame_positions = set()
    
    
    start_time = time.time()
    
    try:
        with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            game_count = 0
            
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_count += 1
                    if game_count % 10000 == 0:
                        elapsed = time.time() - start_time
                        games_per_sec = game_count / elapsed
                        print(f"Processed {game_count:,} games ({games_per_sec:.0f} games/sec)")
                        print(f"Positions: Opening {len(opening_positions):,}, "
                              f"Midgame {len(midgame_positions):,}, Endgame {len(endgame_positions):,}")
                    
                    moves_list = list(game.mainline_moves())
                    if len(moves_list) < 20 or len(moves_list) > 200:
                        continue

                    if (len(opening_positions) >= num_positions_per_phase['opening'] and
                        len(midgame_positions) >= num_positions_per_phase['midgame'] and
                        len(endgame_positions) >= num_positions_per_phase['endgame']):
                        print(f"Target reached after {game_count:,} games!")
                        break

                    board = game.board()
                    move_count = 0
                    
                    for move in moves_list:
                        board.push(move)
                        move_count += 1

                        if move_count % 3 == 0:
                            fen = board.fen()
                            
                            if 1 <= move_count <= 15 and len(opening_positions) < num_positions_per_phase['opening']:
                                opening_positions.add(fen)
                            elif 16 <= move_count <= 35 and len(midgame_positions) < num_positions_per_phase['midgame']:
                                midgame_positions.add(fen)
                            elif move_count >= 36 and len(endgame_positions) < num_positions_per_phase['endgame']:
                                endgame_positions.add(fen)
                
                except Exception as e:
                    print(f"Error processing game {game_count}: {e}")
                    continue 
                    
    except Exception as e:
        print(f"File reading error: {e}")
        return [], [], []
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f} seconds")
    print(f"Processed {game_count:,} games ({game_count/total_time:.0f} games/sec)")
    
    return list(opening_positions), list(midgame_positions), list(endgame_positions)

def evaluate_batch(fen_batch, stockfish_path, batch_id):
    fish = stockfish.Stockfish(stockfish_path)
    fish.set_depth(12)
    
    results = []
    print(f"Batch {batch_id}: Processing {len(fen_batch)} positions...")
    
    for i, fen in enumerate(fen_batch):
        try:
            fish.set_fen_position(fen)
            evaluation = fish.get_evaluation()
            
            if evaluation:
                if evaluation['type'] == 'cp':
                    # Convert centipawns to normalized score
                    normalized_score = np.tanh(evaluation['value'] / 400.0)
                elif evaluation['type'] == 'mate':
                    normalized_score = 1.0 if evaluation['value'] > 0 else -1.0
                else:
                    normalized_score = 0.0
                    
                results.append([fen, normalized_score])
            else:
                continue
                
        except Exception as e:
            print(f"Error evaluating position {i} in batch {batch_id}: {e}")
            continue
    
    print(f"Batch {batch_id}: Completed {len(results)} evaluations")
    return results

def save_positions_to_csv_fast(positions, filename, num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count() - 1, 8) 
    
    stockfish_path = ""
    
    print(f"Starting evaluation of {len(positions)} positions using {num_workers} workers...")
    start_time = time.time()
    
    batch_size = len(positions) // num_workers
    if batch_size < 100:
        batch_size = 100
    
    batches = [positions[i:i + batch_size] for i in range(0, len(positions), batch_size)]
    
    print(f"Created {len(batches)} batches of ~{batch_size} positions each")
    
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(evaluate_batch, batch, stockfish_path, i): i 
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                completed_positions = len(all_results)
                elapsed_time = time.time() - start_time
                positions_per_sec = completed_positions / elapsed_time if elapsed_time > 0 else 0
                eta = (len(positions) - completed_positions) / positions_per_sec if positions_per_sec > 0 else 0
                
                print(f"Progress: {completed_positions}/{len(positions)} positions "
                      f"({positions_per_sec:.1f} pos/sec, ETA: {eta/60:.1f} min)")
                      
            except Exception as e:
                print(f"Batch {batch_id} failed: {e}")
    print("Writing results to CSV")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fen', 'evaluation'])
        writer.writerows(all_results)
    
    total_time = time.time() - start_time
    print(f"Completed Saved {len(all_results)} positions to {filename}")
    print(f"Total time: {total_time/60:.1f} minutes ({len(all_results)/total_time:.1f} positions/second)")
    
    return len(all_results)


def main():
    pgn_file = ""
    
    target_positions = {
        'opening': 50000,
        'midgame': 300000, 
        'endgame': 200000
    }
    
    print("Extracting positions from PGN")
    opening_pos, midgame_pos, endgame_pos = extract_positions_from_pgn(
        pgn_file, target_positions
    )
    
    opening_pos = list(set(opening_pos))
    midgame_pos = list(set(midgame_pos))
    endgame_pos = list(set(endgame_pos))
    
    random.shuffle(opening_pos)
    random.shuffle(midgame_pos)
    random.shuffle(endgame_pos)

    save_positions_to_csv_fast(opening_pos[:target_positions['opening']], 'opening_positions.csv')
    save_positions_to_csv_fast(midgame_pos[:target_positions['midgame']], 'midgame_positions.csv')
    save_positions_to_csv_fast(endgame_pos[:target_positions['endgame']], 'endgame_positions.csv')

    print(f"\nExtracted:")
    print(f"Opening: {len(opening_pos)} positions")
    print(f"Midgame: {len(midgame_pos)} positions") 
    print(f"Endgame: {len(endgame_pos)} positions")

if __name__ == "__main__":
    main()