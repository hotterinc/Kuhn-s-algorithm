import random

def write_graph(filename, n, extras=12, seed=42, perfect=True, s_ratio=0.35, t_ratio=0.25):
    """
    Генерирует двудольный неориентированный граф на 2n вершинах:
      левая доля: 0..n-1, правая: n..2n-1.
    Матрица: '-' = нет ребра, '1' = ребро.

    perfect=True  -> гарантированно есть совершенное паросочетание.
    perfect=False -> нарушаем Холла (нет совершенного).
    extras — плотность дополнительных рёбер от каждой левой вершины.
    """
    N = 2 * n
    matrix = [['-' for _ in range(N)] for _ in range(N)]

    def add_edge(u, v):
        if u == v:
            return
        matrix[u][v] = '1'
        matrix[v][u] = '1'

    L = list(range(n))
    R = list(range(n, 2 * n))
    rng = random.Random(seed)

    if perfect:
        # Базовое совершенное паросочетание: i <-> n+i
        for i in range(n):
            add_edge(L[i], R[i])
    else:
        # Нарушаем Холла: берём S слева (|S|=s) и T справа (|T|=t<s)
        s = max(2, int(n * s_ratio))
        t = max(1, int(n * t_ratio))
        t = min(t, s - 1)

        S = L[:s]
        T = R[:t]

        for u in S:
            for v in T:
                add_edge(u, v)

        remaining_L = L[s:]
        remaining_R = R[t:]
        m = min(len(remaining_L), len(remaining_R))
        for i in range(m):
            add_edge(remaining_L[i], remaining_R[i])

    # Дополнительные рёбра для плотности
    for u in L:
        baseline = set()
        if perfect:
            baseline.add(u)
        choices = [j for j in range(n) if j not in baseline]
        k = min(extras, len(choices))
        for j in rng.sample(choices, k):
            add_edge(u, R[j])

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(N) + '\n')
        for i in range(N):
            f.write(' '.join(matrix[i]) + '\n')

    print(f"Wrote {filename}: N={N}, perfect={perfect}, extras={extras}")

if __name__ == '__main__':
    write_graph('big_perfect_400.txt',   n=200, extras=12, seed=1, perfect=True)
    write_graph('big_noperfect_400.txt', n=200, extras=12, seed=2, perfect=False, s_ratio=0.35, t_ratio=0.25)
