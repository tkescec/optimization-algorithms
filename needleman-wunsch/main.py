# Task 13. The Needleman-Wunch Algorithm
import math
import numpy as np

def initialize_matrix(len_s1, len_s2, gap_penalty):
    matrix = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
    # Initialize first column
    for i in range(1, len_s1 + 1):
        matrix[i][0] = matrix[i-1][0] + gap_penalty
    # Initialize first row
    for j in range(1, len_s2 + 1):
        matrix[0][j] = matrix[0][j-1] + gap_penalty
    return matrix


def fill_matrix(matrix, S1, S2, match_score, mismatch_penalty, gap_penalty):
    len_s1, len_s2 = len(S1), len(S2)
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if S1[i-1] == S2[j-1]:
                score_diag = matrix[i-1][j-1] + match_score
            else:
                score_diag = matrix[i-1][j-1] + mismatch_penalty
            score_up = matrix[i-1][j] + gap_penalty
            score_left = matrix[i][j-1] + gap_penalty
            matrix[i][j] = max(score_diag, score_up, score_left)
    return matrix


def traceback(matrix, S1, S2, match_score, mismatch_penalty, gap_penalty):
    aligned_S1 = []
    aligned_S2 = []
    i, j = len(S1), len(S2)

    while i > 0 and j > 0:
        current_score = matrix[i][j]
        diagonal_score = matrix[i - 1][j - 1]
        up_score = matrix[i - 1][j]
        left_score = matrix[i][j - 1]

        if S1[i - 1] == S2[j - 1]:
            score_diag = diagonal_score + match_score
        else:
            score_diag = diagonal_score + mismatch_penalty

        if current_score == score_diag:
            aligned_S1.append(S1[i - 1])
            aligned_S2.append(S2[j - 1])
            i -= 1
            j -= 1
        elif current_score == up_score + gap_penalty:
            aligned_S1.append(S1[i - 1])
            aligned_S2.append('-')
            i -= 1
        else:
            aligned_S1.append('-')
            aligned_S2.append(S2[j - 1])
            j -= 1

    # Fill the remaining gaps
    while i > 0:
        aligned_S1.append(S1[i - 1])
        aligned_S2.append('-')
        i -= 1
    while j > 0:
        aligned_S1.append('-')
        aligned_S2.append(S2[j - 1])
        j -= 1

    # Reverse the alignments as we built them backwards
    aligned_S1 = ''.join(reversed(aligned_S1))
    aligned_S2 = ''.join(reversed(aligned_S2))

    return aligned_S1, aligned_S2


def needleman_wunsch(S1, S2, match_score, mismatch_penalty, gap_penalty):
    len_s1, len_s2 = len(S1), len(S2)
    matrix = initialize_matrix(len_s1, len_s2, gap_penalty)
    matrix = fill_matrix(matrix, S1, S2, match_score, mismatch_penalty, gap_penalty)
    aligned_S1, aligned_S2 = traceback(matrix, S1, S2, match_score, mismatch_penalty, gap_penalty)
    return matrix, aligned_S1, aligned_S2

# Example sequences
S1 = "ACTGACTGAACCCAA"
S2 = "ACTGATCAA"

# Scoring parameters
match_score = 1
mismatch_penalty = -1
gap_penalty = -2

# Perform alignment
matrix, aligned_S1, aligned_S2 = needleman_wunsch(S1, S2, match_score, mismatch_penalty, gap_penalty)

print("Scoring Matrix:")
print(matrix)
print("\nAligned Sequences:")
print(aligned_S1)
print(aligned_S2)

# Task 14. Time complexity of The Needleman-Wunch Algorithm
# Theory Complexity
# Matrix Initialization: O(m * n)
# Two nested loops are used to initialize the matrix.
# Traceback: O(m + n)
# The traceback step requires traversing the matrix diagonally.


# Scoring parameters
match_score = 1
mismatch_penalty = -1
gap_penalty = 20

# Perform alignment
matrix, aligned_S1, aligned_S2 = needleman_wunsch(S1, S2, match_score, mismatch_penalty, gap_penalty)

print("Scoring Matrix:")
print(matrix)
print("\nAligned Sequences:")
print(aligned_S1)
print(aligned_S2)

