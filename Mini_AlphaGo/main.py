import time


ALL_BITS = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F
INF = 10 ** 9
POPCOUNT_8 = tuple(bin(i).count("1") for i in range(256))

CORNERS = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)
X_SQUARES = (1 << 9) | (1 << 14) | (1 << 49) | (1 << 54)
C_SQUARES = (
    (1 << 1) | (1 << 8) | (1 << 6) | (1 << 15) |
    (1 << 48) | (1 << 57) | (1 << 55) | (1 << 62)
)
EDGES = 0xFF818181818181FF
CORNER_NEIGHBORS = (
    (1 << 0, (1 << 1) | (1 << 8) | (1 << 9)),
    (1 << 7, (1 << 6) | (1 << 14) | (1 << 15)),
    (1 << 56, (1 << 48) | (1 << 49) | (1 << 57)),
    (1 << 63, (1 << 54) | (1 << 55) | (1 << 62)),
)

POSITION_WEIGHTS = (
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -50, -2, -2, -2, -2, -50, -20,
    20, -2, 15, 3, 3, 15, -2, 20,
    5, -2, 3, 3, 3, 3, -2, 5,
    5, -2, 3, 3, 3, 3, -2, 5,
    20, -2, 15, 3, 3, 15, -2, 20,
    -20, -50, -2, -2, -2, -2, -50, -20,
    120, -20, 20, 5, 5, 20, -20, 120,
)


class SearchTimeout(Exception):
    pass


def _iter_bits(bb):
    while bb:
        move = bb & -bb
        yield move
        bb ^= move


def _get_moves(my, opp):
    empty = ALL_BITS ^ (my | opp)
    moves = 0

    m = NOT_A
    c = ((my << 1) & m) & opp
    c |= ((c << 1) & m) & opp
    c |= ((c << 1) & m) & opp
    c |= ((c << 1) & m) & opp
    c |= ((c << 1) & m) & opp
    c |= ((c << 1) & m) & opp
    moves |= ((c << 1) & m) & empty

    m = NOT_H
    c = ((my >> 1) & m) & opp
    c |= ((c >> 1) & m) & opp
    c |= ((c >> 1) & m) & opp
    c |= ((c >> 1) & m) & opp
    c |= ((c >> 1) & m) & opp
    c |= ((c >> 1) & m) & opp
    moves |= ((c >> 1) & m) & empty

    c = (my << 8) & opp
    c |= (c << 8) & opp
    c |= (c << 8) & opp
    c |= (c << 8) & opp
    c |= (c << 8) & opp
    c |= (c << 8) & opp
    moves |= (c << 8) & empty

    c = (my >> 8) & opp
    c |= (c >> 8) & opp
    c |= (c >> 8) & opp
    c |= (c >> 8) & opp
    c |= (c >> 8) & opp
    c |= (c >> 8) & opp
    moves |= (c >> 8) & empty

    m = NOT_A
    c = ((my << 9) & m) & opp
    c |= ((c << 9) & m) & opp
    c |= ((c << 9) & m) & opp
    c |= ((c << 9) & m) & opp
    c |= ((c << 9) & m) & opp
    c |= ((c << 9) & m) & opp
    moves |= ((c << 9) & m) & empty

    m = NOT_H
    c = ((my << 7) & m) & opp
    c |= ((c << 7) & m) & opp
    c |= ((c << 7) & m) & opp
    c |= ((c << 7) & m) & opp
    c |= ((c << 7) & m) & opp
    c |= ((c << 7) & m) & opp
    moves |= ((c << 7) & m) & empty

    m = NOT_A
    c = ((my >> 7) & m) & opp
    c |= ((c >> 7) & m) & opp
    c |= ((c >> 7) & m) & opp
    c |= ((c >> 7) & m) & opp
    c |= ((c >> 7) & m) & opp
    c |= ((c >> 7) & m) & opp
    moves |= ((c >> 7) & m) & empty

    m = NOT_H
    c = ((my >> 9) & m) & opp
    c |= ((c >> 9) & m) & opp
    c |= ((c >> 9) & m) & opp
    c |= ((c >> 9) & m) & opp
    c |= ((c >> 9) & m) & opp
    c |= ((c >> 9) & m) & opp
    moves |= ((c >> 9) & m) & empty

    return moves & ALL_BITS


def _do_move(my, opp, move):
    flipped = 0

    f = 0
    p = (move << 1) & NOT_A
    while p & opp:
        f |= p
        p = (p << 1) & NOT_A
    if p & my:
        flipped |= f

    f = 0
    p = (move >> 1) & NOT_H
    while p & opp:
        f |= p
        p = (p >> 1) & NOT_H
    if p & my:
        flipped |= f

    f = 0
    p = (move << 8) & ALL_BITS
    while p & opp:
        f |= p
        p = (p << 8) & ALL_BITS
    if p & my:
        flipped |= f

    f = 0
    p = move >> 8
    while p & opp:
        f |= p
        p >>= 8
    if p & my:
        flipped |= f

    f = 0
    p = (move << 9) & NOT_A & ALL_BITS
    while p & opp:
        f |= p
        p = (p << 9) & NOT_A & ALL_BITS
    if p & my:
        flipped |= f

    f = 0
    p = (move << 7) & NOT_H & ALL_BITS
    while p & opp:
        f |= p
        p = (p << 7) & NOT_H & ALL_BITS
    if p & my:
        flipped |= f

    f = 0
    p = (move >> 7) & NOT_A
    while p & opp:
        f |= p
        p = (p >> 7) & NOT_A
    if p & my:
        flipped |= f

    f = 0
    p = (move >> 9) & NOT_H
    while p & opp:
        f |= p
        p = (p >> 9) & NOT_H
    if p & my:
        flipped |= f

    return (my | move | flipped) & ALL_BITS, opp & ~flipped


def _board_to_bits(board):
    black = 0
    white = 0
    for row in range(8):
        base = row << 3
        for col, piece in enumerate(board._board[row]):
            bit = 1 << (base + col)
            if piece == "X":
                black |= bit
            elif piece == "O":
                white |= bit
    return black, white


def _bit_to_coord(bit):
    idx = bit.bit_length() - 1
    row = idx >> 3
    col = idx & 7
    return chr(ord("A") + col) + str(row + 1)


def _adjacent(bb):
    return (
        ((bb << 1) & NOT_A) |
        ((bb >> 1) & NOT_H) |
        ((bb << 8) & ALL_BITS) |
        (bb >> 8) |
        ((bb << 9) & NOT_A & ALL_BITS) |
        ((bb << 7) & NOT_H & ALL_BITS) |
        ((bb >> 7) & NOT_A) |
        ((bb >> 9) & NOT_H)
    ) & ALL_BITS


def _scaled_diff(a, b):
    total = a + b
    if total == 0:
        return 0
    return 100 * (a - b) // total


def _popcount(x):
    return (
        POPCOUNT_8[x & 0xFF] +
        POPCOUNT_8[(x >> 8) & 0xFF] +
        POPCOUNT_8[(x >> 16) & 0xFF] +
        POPCOUNT_8[(x >> 24) & 0xFF] +
        POPCOUNT_8[(x >> 32) & 0xFF] +
        POPCOUNT_8[(x >> 40) & 0xFF] +
        POPCOUNT_8[(x >> 48) & 0xFF] +
        POPCOUNT_8[(x >> 56) & 0xFF]
    )


def _positional_score(my, opp):
    score = 0
    for bb, sign in ((my, 1), (opp, -1)):
        current = bb
        while current:
            bit = current & -current
            score += sign * POSITION_WEIGHTS[bit.bit_length() - 1]
            current ^= bit
    return score


class AIPlayer:
    def __init__(self, color):
        self.color = color
        self.max_depth = 12
        self.tt = {}
        self.deadline = 0.0
        self.nodes = 0
        self.best_move = 0

    def _time_budget(self, empties):
        if empties > 50:
            return 1.2
        if empties > 40:
            return 2.2
        if empties > 28:
            return 3.5
        if empties > 16:
            return 4.2
        if empties > 10:
            return 4.8
        return 8.0

    def _solve_threshold(self, empties, legal_count):
        if empties <= 14 and legal_count <= 6:
            return 14
        return 12

    def _depth_limit(self, empties, solve_threshold):
        if empties <= solve_threshold:
            return empties
        return self.max_depth

    def get_move(self, board):
        player_name = "黑棋" if self.color == "X" else "白棋"
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        black, white = _board_to_bits(board)
        my = black if self.color == "X" else white
        opp = white if self.color == "X" else black
        legal_moves = _get_moves(my, opp)
        if legal_moves == 0:
            return None
        if legal_moves & (legal_moves - 1) == 0:
            return _bit_to_coord(legal_moves)

        corner_moves = legal_moves & CORNERS
        if corner_moves:
            corner_list = self._order_moves(my, opp, corner_moves, 0)
            self.best_move = corner_list[0]
            return _bit_to_coord(corner_list[0])

        empties = 64 - _popcount(my | opp)
        legal_count = _popcount(legal_moves)
        solve_threshold = self._solve_threshold(empties, legal_count)
        budget = min(self._time_budget(empties), 55.0)
        self.deadline = time.perf_counter() + budget
        self.nodes = 0
        self.tt.clear()

        ordered_moves = self._order_moves(my, opp, legal_moves, 0)
        best_move = ordered_moves[0]
        completed_depth = 0

        try:
            max_depth = min(self._depth_limit(empties, solve_threshold), empties)
            for depth in range(1, max_depth + 1):
                move, _ = self._search_root(my, opp, depth, best_move)
                if move:
                    best_move = move
                    completed_depth = depth
                if empties <= solve_threshold and completed_depth >= empties:
                    break
        except SearchTimeout:
            pass

        self.best_move = best_move
        return _bit_to_coord(best_move)

    def _check_time(self):
        self.nodes += 1
        if self.nodes & 1023 == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def _terminal_score(self, my, opp):
        diff = _popcount(my) - _popcount(opp)
        if diff > 0:
            return 1000000 + diff
        if diff < 0:
            return -1000000 + diff
        return 0

    def _corner_pressure(self, my, opp, occupied):
        score = 0
        for corner, neighbors in CORNER_NEIGHBORS:
            if occupied & corner:
                continue
            score += _popcount(opp & neighbors) - _popcount(my & neighbors)
        return score

    def _frontier_score(self, my, opp, occupied):
        empty = ALL_BITS ^ occupied
        frontier = _adjacent(empty)
        my_frontier = _popcount(my & frontier)
        opp_frontier = _popcount(opp & frontier)
        return _scaled_diff(opp_frontier, my_frontier)

    def _evaluate(self, my, opp, my_moves):
        occupied = my | opp
        empties = 64 - _popcount(occupied)
        opp_moves = _get_moves(opp, my)

        mobility = _scaled_diff(_popcount(my_moves), _popcount(opp_moves))
        disc_diff = _scaled_diff(_popcount(my), _popcount(opp))
        positional = _positional_score(my, opp)
        corners = 25 * (_popcount(my & CORNERS) - _popcount(opp & CORNERS))
        edges = _popcount(my & EDGES) - _popcount(opp & EDGES)
        frontier = self._frontier_score(my, opp, occupied)
        corner_pressure = self._corner_pressure(my, opp, occupied)
        parity = 1 if empties & 1 else -1

        if empties > 40:
            return (
                10 * mobility +
                20 * corners +
                12 * corner_pressure +
                4 * positional +
                3 * frontier +
                edges
            )

        if empties > 16:
            return (
                8 * mobility +
                30 * corners +
                10 * corner_pressure +
                3 * frontier +
                3 * positional +
                2 * edges +
                2 * disc_diff
            )

        return (
            5 * mobility +
            35 * corners +
            4 * corner_pressure +
            4 * frontier +
            2 * positional +
            4 * edges +
            12 * disc_diff +
            8 * parity
        )

    def _order_moves(self, my, opp, moves, tt_move):
        ordered = []
        for move in _iter_bits(moves):
            idx = move.bit_length() - 1
            score = POSITION_WEIGHTS[idx] * 8
            if move & CORNERS:
                score += 200000
            if move == tt_move:
                score += 100000
            if move & X_SQUARES:
                score -= 6000
            elif move & C_SQUARES:
                score -= 2500

            new_my, new_opp = _do_move(my, opp, move)
            score -= _popcount(_get_moves(new_opp, new_my)) * 80
            score += (_popcount(new_my) - _popcount(my)) * 8
            ordered.append((score, move))

        ordered.sort(reverse=True)
        return [move for _, move in ordered]

    def _search_root(self, my, opp, depth, preferred_move):
        legal_moves = _get_moves(my, opp)
        ordered_moves = self._order_moves(my, opp, legal_moves, preferred_move)
        alpha = -INF
        beta = INF
        best_score = -INF
        best_move = ordered_moves[0]

        for i, move in enumerate(ordered_moves):
            self._check_time()
            new_my, new_opp = _do_move(my, opp, move)
            if i == 0:
                score = -self._search(new_opp, new_my, depth - 1, -beta, -alpha, False)
            else:
                score = -self._search(new_opp, new_my, depth - 1, -alpha - 1, -alpha, False)
                if alpha < score < beta:
                    score = -self._search(new_opp, new_my, depth - 1, -beta, -score, False)
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_move, best_score

    def _search(self, my, opp, depth, alpha, beta, passed):
        self._check_time()
        alpha_orig = alpha
        key = (my, opp, passed)
        entry = self.tt.get(key)
        tt_move = 0

        if entry is not None:
            stored_depth, flag, value, tt_move = entry
            if stored_depth >= depth:
                if flag == 0:
                    return value
                if flag == 1 and value > alpha:
                    alpha = value
                elif flag == 2 and value < beta:
                    beta = value
                if alpha >= beta:
                    return value

        legal_moves = _get_moves(my, opp)
        if legal_moves == 0:
            if passed:
                return self._terminal_score(my, opp)
            value = -self._search(opp, my, depth, -beta, -alpha, True)
            self.tt[key] = (depth, 0, value, 0)
            return value

        occupied = my | opp
        empties = 64 - _popcount(occupied)
        solve_threshold = self._solve_threshold(empties, _popcount(legal_moves))
        if depth <= 0 and empties > solve_threshold:
            return self._evaluate(my, opp, legal_moves)

        best_score = -INF
        best_move = 0
        ordered_moves = self._order_moves(my, opp, legal_moves, tt_move)

        for i, move in enumerate(ordered_moves):
            new_my, new_opp = _do_move(my, opp, move)
            if i == 0:
                score = -self._search(new_opp, new_my, depth - 1, -beta, -alpha, False)
            else:
                score = -self._search(new_opp, new_my, depth - 1, -alpha - 1, -alpha, False)
                if alpha < score < beta:
                    score = -self._search(new_opp, new_my, depth - 1, -beta, -score, False)
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = 0
        if best_score <= alpha_orig:
            flag = 2
        elif best_score >= beta:
            flag = 1
        self.tt[key] = (depth, flag, best_score, best_move)
        return best_score
