"""
[1,1,1,1,0,0,0]
[1,1,1,1,1,0,0]
[1,1,1,1,1,1,0]
[0,1,1,1,1,1,1]
[0,0,1,1,1,1,1]
[0,0,0,1,1,1,1]
"""


class Board:
    def __init__(self, board):
        self.board = board
        self.real_player = -1
        self.robot_player = 1
        self.empty = 0

    def add_piece(self, column, real_player=True):
        for row in range(0, 5):
            print(row)
            if self.board[row][column] == self.empty:
                if real_player:
                    self.board[row][column] = self.real_player
                else:
                    self.board[row][column] = self.robot_player
                return True
        return False

    def display_board(self):
        for row in self.board[::-1]:
            print(row)

    def terminal_state(self):
        terminal, player = self.check_horizontal()
        print(terminal, player)
        terminal, player = self.check_vertical()
        print(terminal, player)

        # check all possible (only at least 4 long) diagonal from bottom left corner to right top corner
        for value in range(0,4):
          terminal, player = self.diagonal(value, 0, going_up=True)
        for value in range(1,4):
          terminal, player = self.diagonal(0, value, going_up=True)

        for value in range(0,4):
            terminal, player = self.diagonal(5, value, going_up=False)
        for value in range(6,3,-1):
            terminal, player = self.diagonal(value, 0, going_up=False)

    def consecutive(self, cell, ai, consecutive):
        if cell == self.robot_player:
            if ai:
                consecutive += 1
            else:
                ai = True
                consecutive = 1
        elif cell == self.real_player:
            if not ai:
                consecutive += 1
            else:
                ai = False
                consecutive = 1

        return ai, consecutive

    def check_horizontal(self):
        ai = True
        for row in self.board:
            consecutive = 0  # note consecutive same type
            for col in row:
                ai, consecutive = self.consecutive(col, ai, consecutive)

                if consecutive == 4:
                    return True, ai

        return False, ai

    def check_vertical(self):
        ai = True
        for column in range(0, 7):
            consecutive = 0
            for row in self.board:
                ai, consecutive = self.consecutive(row[column], ai, consecutive)
                if consecutive == 4:
                    return True, ai

        return False, ai

    def diagonal(self, starting_row, starting_column, going_up=True):
        ai = True
        consecutive = 0
        #value = 4
        while True:
            try:
                if starting_row >= len(self.board) or starting_column >= len(self.board[0]) or starting_row < 0 or starting_column <0:
                  return False, ai

                #value += 1
                ai, consecutive = self.consecutive(self.board[starting_row][starting_column], ai, consecutive)

                if consecutive == 4:
                  return True, ai

                if going_up:
                    starting_row += 1
                else:
                    starting_row -= 1

                starting_column += 1

            except:
                return False, ai


board = []
for x in range(6):
    board.append([0] * 7)

connect4 = Board(board)

connect4.add_piece(0, real_player=False)
connect4.add_piece(5, real_player=True)
connect4.add_piece(5, real_player=True)
connect4.add_piece(5, real_player=True)
connect4.add_piece(5, real_player=True)

connect4.terminal_state()

connect4.display_board()
