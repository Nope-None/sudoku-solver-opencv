def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None


def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in [board[r][col] for r in range(9)]:
        return False
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if board[r][c] == num:
                return False
    return True


def solve(board):
    empty = find_empty(board)
    if empty is None:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve(board):
                return True
            board[row][col] = 0
    return False


def print_board(board):
    print("+" + "-------+" * 3)
    for i, row in enumerate(board):
        if i > 0 and i % 3 == 0:
            print("+" + "-------+" * 3)
        row_str = "| "
        for j, val in enumerate(row):
            row_str += (str(val) if val != 0 else ".") + " "
            if (j + 1) % 3 == 0:
                row_str += "| "
        print(row_str)
    print("+" + "-------+" * 3)
