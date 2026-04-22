import time


FULL_BOARD = (1 << 64) - 1
KEEP_FROM_A = 0xFEFEFEFEFEFEFEFE
KEEP_FROM_H = 0x7F7F7F7F7F7F7F7F
WIN_SCORE = 10 ** 6
SEARCH_INF = 10 ** 9
BYTE_COUNTS = tuple(bin(i).count("1") for i in range(256))

CORNER_MASK = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)
DIAGONAL_DANGER = (1 << 9) | (1 << 14) | (1 << 49) | (1 << 54)
EDGE_DANGER = (
    (1 << 1) | (1 << 8) | (1 << 6) | (1 << 15) |
    (1 << 48) | (1 << 57) | (1 << 55) | (1 << 62)
)
EDGE_MASK = 0xFF818181818181FF
OPEN_CORNER_AREAS = (
    (1 << 0, (1 << 1) | (1 << 8) | (1 << 9)),
    (1 << 7, (1 << 6) | (1 << 14) | (1 << 15)),
    (1 << 56, (1 << 48) | (1 << 49) | (1 << 57)),
    (1 << 63, (1 << 54) | (1 << 55) | (1 << 62)),
)

SQUARE_TABLE = (
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -50, -2, -2, -2, -2, -50, -20,
    20, -2, 15, 3, 3, 15, -2, 20,
    5, -2, 3, 3, 3, 3, -2, 5,
    5, -2, 3, 3, 3, 3, -2, 5,
    20, -2, 15, 3, 3, 15, -2, 20,
    -20, -50, -2, -2, -2, -2, -50, -20,
    120, -20, 20, 5, 5, 20, -20, 120,
)

DIRECTIONS = (
    (1, KEEP_FROM_A),
    (-1, KEEP_FROM_H),
    (8, FULL_BOARD),
    (-8, FULL_BOARD),
    (9, KEEP_FROM_A),
    (7, KEEP_FROM_H),
    (-7, KEEP_FROM_A),
    (-9, KEEP_FROM_H),
)


class SearchTimeout(Exception):
    pass


def _shift(bits, step, file_mask):
    if step > 0:
        return ((bits << step) & file_mask) & FULL_BOARD
    return (bits >> -step) & file_mask


def _each_bit(mask):
    while mask:
        low_bit = mask & -mask
        yield low_bit
        mask ^= low_bit


def _legal_moves(us, them):
    blank = FULL_BOARD ^ (us | them)
    choices = 0
    for step, file_mask in DIRECTIONS:
        captured_line = _shift(us, step, file_mask) & them
        for _ in range(5):
            captured_line |= _shift(captured_line, step, file_mask) & them
        choices |= _shift(captured_line, step, file_mask) & blank
    return choices & FULL_BOARD


def _play(us, them, move):
    flipped = 0
    for step, file_mask in DIRECTIONS:
        ray = 0
        cursor = _shift(move, step, file_mask)
        while cursor & them:
            ray |= cursor
            cursor = _shift(cursor, step, file_mask)
        if cursor & us:
            flipped |= ray
    return (us | move | flipped) & FULL_BOARD, them & ~flipped


def _board_to_bitboards(board):
    dark = 0
    light = 0
    for r in range(8):
        row_offset = r << 3
        for c, stone in enumerate(board._board[r]):
            place = 1 << (row_offset + c)
            if stone == "X":
                dark |= place
            elif stone == "O":
                light |= place
    return dark, light


def _move_name(move):
    index = move.bit_length() - 1
    return chr(ord("A") + (index & 7)) + str((index >> 3) + 1)


def _neighbor_cells(mask):
    around = 0
    for step, file_mask in DIRECTIONS:
        around |= _shift(mask, step, file_mask)
    return around & FULL_BOARD


def _ratio_diff(left, right):
    total = left + right
    return 0 if total == 0 else 100 * (left - right) // total


def _count_bits(value):
    return (
        BYTE_COUNTS[value & 0xFF] +
        BYTE_COUNTS[(value >> 8) & 0xFF] +
        BYTE_COUNTS[(value >> 16) & 0xFF] +
        BYTE_COUNTS[(value >> 24) & 0xFF] +
        BYTE_COUNTS[(value >> 32) & 0xFF] +
        BYTE_COUNTS[(value >> 40) & 0xFF] +
        BYTE_COUNTS[(value >> 48) & 0xFF] +
        BYTE_COUNTS[(value >> 56) & 0xFF]
    )


def _weight_sum(us, them):
    value = 0
    for stones, sign in ((us, 1), (them, -1)):
        rest = stones
        while rest:
            bit = rest & -rest
            value += sign * SQUARE_TABLE[bit.bit_length() - 1]
            rest ^= bit
    return value


class AIPlayer:
    def __init__(self, color):
        self.color = color
        self.max_depth = 12
        self.transposition = {}
        self.stop_at = 0.0
        self.visited_nodes = 0
        self.best_move = 0

    def _move_time_limit(self, empty_count):
        if empty_count > 50:
            return 1.2
        if empty_count > 40:
            return 2.2
        if empty_count > 28:
            return 3.5
        if empty_count > 16:
            return 4.2
        if empty_count > 10:
            return 4.8
        return 8.0

    def _exact_limit(self, empty_count, branch_count):
        if empty_count <= 14 and branch_count <= 6:
            return 14
        return 12

    def _iterative_limit(self, empty_count, exact_limit):
        if empty_count <= exact_limit:
            return empty_count
        return self.max_depth

    def get_move(self, board):
        side_name = "黑棋" if self.color == "X" else "白棋"
        print("请等一会，对方 {}-{} 正在思考中...".format(side_name, self.color))

        black, white = _board_to_bitboards(board)
        us = black if self.color == "X" else white
        them = white if self.color == "X" else black
        legal = _legal_moves(us, them)
        if legal == 0:
            return None
        if legal & (legal - 1) == 0:
            return _move_name(legal)

        forced_corners = legal & CORNER_MASK
        if forced_corners:
            corner_order = self._rank_moves(us, them, forced_corners, 0)
            self.best_move = corner_order[0]
            return _move_name(corner_order[0])

        empty_count = 64 - _count_bits(us | them)
        branch_count = _count_bits(legal)
        exact_limit = self._exact_limit(empty_count, branch_count)
        self.stop_at = time.perf_counter() + min(self._move_time_limit(empty_count), 55.0)
        self.visited_nodes = 0
        self.transposition.clear()

        ordered = self._rank_moves(us, them, legal, 0)
        chosen = ordered[0]
        finished_depth = 0

        try:
            max_depth = min(self._iterative_limit(empty_count, exact_limit), empty_count)
            for depth in range(1, max_depth + 1):
                candidate, _ = self._root_search(us, them, depth, chosen)
                if candidate:
                    chosen = candidate
                    finished_depth = depth
                if empty_count <= exact_limit and finished_depth >= empty_count:
                    break
        except SearchTimeout:
            pass

        self.best_move = chosen
        return _move_name(chosen)

    def _guard_clock(self):
        self.visited_nodes += 1
        if self.visited_nodes & 1023 == 0 and time.perf_counter() >= self.stop_at:
            raise SearchTimeout

    def _final_score(self, us, them):
        margin = _count_bits(us) - _count_bits(them)
        if margin > 0:
            return WIN_SCORE + margin
        if margin < 0:
            return -WIN_SCORE + margin
        return 0

    def _corner_risk(self, us, them, occupied):
        risk = 0
        for corner, near_corner in OPEN_CORNER_AREAS:
            if occupied & corner:
                continue
            risk += _count_bits(them & near_corner) - _count_bits(us & near_corner)
        return risk

    def _frontier_balance(self, us, them, occupied):
        empty = FULL_BOARD ^ occupied
        frontier = _neighbor_cells(empty)
        our_frontier = _count_bits(us & frontier)
        their_frontier = _count_bits(them & frontier)
        return _ratio_diff(their_frontier, our_frontier)

    def _evaluate(self, us, them, legal):
        occupied = us | them
        empty_count = 64 - _count_bits(occupied)
        reply = _legal_moves(them, us)

        mobility = _ratio_diff(_count_bits(legal), _count_bits(reply))
        disc_delta = _ratio_diff(_count_bits(us), _count_bits(them))
        square_score = _weight_sum(us, them)
        corners = 25 * (_count_bits(us & CORNER_MASK) - _count_bits(them & CORNER_MASK))
        edge_score = _count_bits(us & EDGE_MASK) - _count_bits(them & EDGE_MASK)
        frontier = self._frontier_balance(us, them, occupied)
        corner_risk = self._corner_risk(us, them, occupied)
        parity = 1 if empty_count & 1 else -1

        if empty_count > 40:
            return (
                10 * mobility +
                20 * corners +
                12 * corner_risk +
                4 * square_score +
                3 * frontier +
                edge_score
            )

        if empty_count > 16:
            return (
                8 * mobility +
                30 * corners +
                10 * corner_risk +
                3 * frontier +
                3 * square_score +
                2 * edge_score +
                2 * disc_delta
            )

        return (
            5 * mobility +
            35 * corners +
            4 * corner_risk +
            4 * frontier +
            2 * square_score +
            4 * edge_score +
            12 * disc_delta +
            8 * parity
        )

    def _rank_moves(self, us, them, legal, priority_move):
        scored = []
        for move in _each_bit(legal):
            index = move.bit_length() - 1
            value = SQUARE_TABLE[index] * 8
            if move & CORNER_MASK:
                value += 200000
            if move == priority_move:
                value += 100000
            if move & DIAGONAL_DANGER:
                value -= 6000
            elif move & EDGE_DANGER:
                value -= 2500

            next_us, next_them = _play(us, them, move)
            value -= _count_bits(_legal_moves(next_them, next_us)) * 80
            value += (_count_bits(next_us) - _count_bits(us)) * 8
            scored.append((value, move))

        scored.sort(reverse=True)
        return [move for _, move in scored]

    def _root_search(self, us, them, depth, priority_move):
        legal = _legal_moves(us, them)
        ordered = self._rank_moves(us, them, legal, priority_move)
        alpha = -SEARCH_INF
        beta = SEARCH_INF
        best_value = -SEARCH_INF
        best_move = ordered[0]

        for i, move in enumerate(ordered):
            self._guard_clock()
            next_us, next_them = _play(us, them, move)
            if i == 0:
                value = -self._pvs(next_them, next_us, depth - 1, -beta, -alpha, False)
            else:
                value = -self._pvs(next_them, next_us, depth - 1, -alpha - 1, -alpha, False)
                if alpha < value < beta:
                    value = -self._pvs(next_them, next_us, depth - 1, -beta, -value, False)
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value

        return best_move, best_value

    def _pvs(self, us, them, depth, alpha, beta, passed):
        self._guard_clock()
        original_alpha = alpha
        key = (us, them, passed)
        cached = self.transposition.get(key)
        cached_move = 0

        if cached is not None:
            saved_depth, bound_type, saved_value, cached_move = cached
            if saved_depth >= depth:
                if bound_type == 0:
                    return saved_value
                if bound_type == 1 and saved_value > alpha:
                    alpha = saved_value
                elif bound_type == 2 and saved_value < beta:
                    beta = saved_value
                if alpha >= beta:
                    return saved_value

        legal = _legal_moves(us, them)
        if legal == 0:
            if passed:
                return self._final_score(us, them)
            value = -self._pvs(them, us, depth, -beta, -alpha, True)
            self.transposition[key] = (depth, 0, value, 0)
            return value

        occupied = us | them
        empty_count = 64 - _count_bits(occupied)
        exact_limit = self._exact_limit(empty_count, _count_bits(legal))
        if depth <= 0 and empty_count > exact_limit:
            return self._evaluate(us, them, legal)

        best_value = -SEARCH_INF
        best_move = 0
        ordered = self._rank_moves(us, them, legal, cached_move)

        for i, move in enumerate(ordered):
            next_us, next_them = _play(us, them, move)
            if i == 0:
                value = -self._pvs(next_them, next_us, depth - 1, -beta, -alpha, False)
            else:
                value = -self._pvs(next_them, next_us, depth - 1, -alpha - 1, -alpha, False)
                if alpha < value < beta:
                    value = -self._pvs(next_them, next_us, depth - 1, -beta, -value, False)
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        bound_type = 0
        if best_value <= original_alpha:
            bound_type = 2
        elif best_value >= beta:
            bound_type = 1
        self.transposition[key] = (depth, bound_type, best_value, best_move)
        return best_value
