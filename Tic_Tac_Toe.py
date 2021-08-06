#Evaluation function
def check_state(board):
    for x in range(3):
        if sum(board[[x,x+3,x+6]]) == 3 or sum(board[x*3:(x*3)+3]) == 3:
            return 1
        elif sum(board[[x,x+3,x+6]]) == -3 or sum(board[x*3:(x*3)+3]) == -3:
            return -1
    if sum(board[[0,4,8]]) == 3 or sum(board[[2,4,6]]) == 3:
        return 1
    elif sum(board[[0,4,8]]) == -3 or sum(board[[2,4,6]]) == -3:
        return -1
    else:
        return 0

def possible_moves(board,p_value):
    branch = []
    for x in range(len(board)):
        if board[x] == 0:
            b_hold = deepcopy(board)
            b_hold[x] = p_value
            branch.append(b_hold)
    return branch

def minimax_search(board,player,max_depth=None,depth=None):
    scores = []
    if (depth is None and max_depth is None) or depth <= max_depth:
        for m in possible_moves(board,player):
            state = check_state(m)
            if state == 0:
                scores.append(minimax_search(m,player*(-1),max_depth=max_depth,depth=depth+1) if max_depth is not None and depth is not None else minimax_search(m,player*(-1)))
            else:
                return state
    if len(scores) == 0:
        scores = [0]
    return max(scores) if player > 0 else min(scores)

def best_move(board,player,max_depth=None):
    result = board
    b_score = float('-inf') if player > 0 else float('inf')
    for m in possible_moves(board,player):
        if check_state(m) == 0:
            score = minimax_search(m,player*(-1),max_depth=max_depth,depth=1) if max_depth is not None else minimax_search(m,player*(-1))
            if (player > 0 and score > b_score) or (player < 0 and score < b_score):
                b_score = score
                result = m
        else:
            return m
    return result

b_map = {0:'*',-1:'X',1:'O'}
board = np.array([0,0,0,0,0,0,0,0,0])
print(
'''
****************************
   WELCOME TO TIC TAC TOE
****************************
'''
)
while True:
    player = input('Would you like to be X or O?\n').upper()
    if player == 'X' or player == 'O':
        break
    else:
        print('Sorry that is not a valid option.\n')
player = -1 if player == 'X' else 1
f_move = random.choice([True, False])
if f_move == True:
    print('\n',np.array([b_map[p] for p in board if p in b_map]).reshape(3,3))
while True:
    if f_move == True:
        p_move = input('\nWhere would you like to play?\n')
    else:
        p_move = 99
    if (int(p_move) < len(board) and board[int(p_move)] == 0) or f_move == False:
        if f_move == True:
            board[int(p_move)] = player
        else:
            f_move = True
        if check_state(board) == 0:
            board = best_move(board,player*(-1))
        print('\n',np.array([b_map[p] for p in board if p in b_map]).reshape(3,3))
        if check_state(board) == player*(-1):
            print('\nYOU LOOSE')
            break
        elif check_state(board) == player:
            print('\nYOU WIN')
            break
        elif 0 not in board:
            print('\nTIE GAME')
            break
    else:
        print('ERROR BAD MOVE SOMEONE IS ALREADY THERE')
