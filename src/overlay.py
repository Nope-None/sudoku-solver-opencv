import cv2
import numpy as np


def draw_solution_on_warped(warped, original_board, solved_board,
                             cell_size=50, font_color=(0, 180, 80),
                             font_scale=1.2, thickness=2):
    output = warped.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in range(9):
        for c in range(9):
            if original_board[r][c] == 0:
                x = c * cell_size + cell_size // 4
                y = r * cell_size + cell_size - cell_size // 6
                cv2.putText(output, str(solved_board[r][c]),
                            (x, y), font, font_scale, font_color, thickness)
    return output


def unwarp_solution(solved_warped, original, M):
    h, w = original.shape[:2]
    warp_h, warp_w = solved_warped.shape[:2]
    M_inv = cv2.invert(M)[1]
    unwarped = cv2.warpPerspective(solved_warped, M_inv, (w, h))
    mask = np.ones((warp_h, warp_w), dtype=np.uint8) * 255
    mask_unwarped = cv2.warpPerspective(mask, M_inv, (w, h))
    mask_3ch = cv2.cvtColor(mask_unwarped, cv2.COLOR_GRAY2BGR)
    result = original.copy()
    result[mask_3ch > 0] = unwarped[mask_3ch > 0]
    return result
