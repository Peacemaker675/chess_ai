# 🤖 Chess AI

A pygame based chess game which uses a CNN trained on 100k chess positions + minimax algorithm with alpha-beta pruning for best play.
You can set the search depth for the model in game.py , the compiled binary has been set to depth = 2 for faster play.

---

## 🎯 Features

- ✅ **Minimax with Alpha‑Beta Pruning** — optimizes search to evaluate the best move efficiently  
- 🎛️ **Configurable Search Depth** — adjust how many plies ahead the AI looks   
- 🛠️ **Python-based** — powered by `python-chess` for board logic  

---

## 🗂️ File Structure
-/src_file:<br>
   1.game.py - main files which contains the logic for gui with pygame and move prediction through the trained model<br>
   2.dataset.py - can be used to create a dataset of randomly generated positons , requires stockfish to label them (used for previous version of CNN)<br>
   3.pgn_to_fen.py - can be used to convert pgn gamefiles into seperate datasets of opening, mid game and end game positions(requires stockfish)<br>

## ⚠️ Requirements

- Python 3.12.8  
- Install dependencies:
```bash
pip install -r requirements.txt
```
## Compiled Binaries
  you can download the bundled exe file from here  - <a href ='https://drive.google.com/file/d/1ZTZupO2g8Ky8WV6ZRV1M-iiPw8qtlPdc/view?usp=drive_link'> LINK </a>
