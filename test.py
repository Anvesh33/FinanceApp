import math

def minimax(depth, node_index, is_maximizing_player, scores, height):
    # Base case: leaf node
    if depth == height:
        return scores[node_index]

    if is_maximizing_player:
        # Maximizer's turn
        return max(
            minimax(depth + 1, node_index * 2, False, scores, height),
            minimax(depth + 1, node_index * 2 + 1, False, scores, height)
        )
    else:
        # Minimizer's turn
        return min(
            minimax(depth + 1, node_index * 2, True, scores, height),
            minimax(depth + 1, node_index * 2 + 1, True, scores, height)
        )

def log2(n):
    return int(math.log2(n))

def print_tree(scores):
    height = log2(len(scores))
    print("\nGame Tree Representation:")
    level = 0
    index = 0
    while index < len(scores):
        count = 2**level
        level_scores = scores[index:index + count]
        spacing = " " * (2**(height - level))
        line = spacing.join(f"{s:2}" for s in level_scores)
        print(" " * (2**(height - level)) + line)
        index += count
        level += 1

if __name__ == "__main__":
    scores = [3, 5, 2, 9, 12, 5, 23, 23]
    height = log2(len(scores))
    print_tree(scores)
    optimal_value = minimax(0, 0, True, scores, height)
    print("\nThe optimal value is:", optimal_value)
