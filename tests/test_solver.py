import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import pytest
from src.solver import solve, find_empty, is_valid

EASY_PUZZLE = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]

EASY_SOLUTION = [
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9],
]

INVALID_PUZZLE = [
    [5,5,0,0,7,0,0,0,0],  # two 5s in same row
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]


class TestFindEmpty:
    def test_finds_first_empty(self):
        board = [[1]*9 for _ in range(9)]
        board[2][3] = 0
        assert find_empty(board) == (2, 3)

    def test_returns_none_when_full(self):
        board = [[1]*9 for _ in range(9)]
        assert find_empty(board) is None


class TestIsValid:
    def test_valid_placement(self):
        board = [[0]*9 for _ in range(9)]
        assert is_valid(board, 0, 0, 5) is True

    def test_invalid_row(self):
        board = [[0]*9 for _ in range(9)]
        board[0][4] = 5
        assert is_valid(board, 0, 0, 5) is False

    def test_invalid_col(self):
        board = [[0]*9 for _ in range(9)]
        board[4][0] = 5
        assert is_valid(board, 0, 0, 5) is False

    def test_invalid_box(self):
        board = [[0]*9 for _ in range(9)]
        board[1][1] = 5
        assert is_valid(board, 0, 0, 5) is False


class TestSolve:
    def test_solves_easy_puzzle(self):
        board = copy.deepcopy(EASY_PUZZLE)
        assert solve(board) is True
        assert board == EASY_SOLUTION

    def test_unsolvable_returns_false(self):
        board = copy.deepcopy(INVALID_PUZZLE)
        assert solve(board) is False

    def test_already_solved_board(self):
        board = copy.deepcopy(EASY_SOLUTION)
        assert solve(board) is True
        assert board == EASY_SOLUTION
