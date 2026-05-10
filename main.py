"""
main.py — OpenCV Sudoku Solver entry point

Usage:
    python main.py --train                              # train digit classifier
    python main.py --image images/sudoku.jpg            # solve & display
    python main.py --image images/sudoku.jpg --output images/solved.jpg
"""

import argparse
import copy
import sys

import cv2

from src.preprocess import preprocess, find_board_contour, warp_board, extract_cells
from src.solver import solve, print_board
from src.overlay import draw_solution_on_warped, unwarp_solution


def load_model(model_path="models/digit_classifier.h5"):
    try:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(model_path)
        print(f"[INFO] Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("        Run:  python main.py --train")
        sys.exit(1)


def run_pipeline(image_path, model_path, output_path=None):
    # 1. Load
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    print(f"[INFO] Loaded: {image_path}  shape={image.shape}")

    # 2. Preprocess & locate board
    thresh = preprocess(image)
    corners = find_board_contour(thresh)
    if corners is None:
        print("[ERROR] Could not locate Sudoku board in image.")
        sys.exit(1)
    print("[INFO] Board detected.")

    warped, M, src = warp_board(image, corners, size=450)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warped_thresh = cv2.threshold(
        warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cells = extract_cells(warped_thresh)

    # 3. Read digits
    from src.digit_recognizer import read_board
    model = load_model(model_path)
    original_board = read_board(cells, model)
    print("\n[INFO] Detected board:")
    print_board(original_board)

    # 4. Solve
    solved_board = copy.deepcopy(original_board)
    if not solve(solved_board):
        print("[ERROR] No solution exists for this board.")
        sys.exit(1)
    print("\n[INFO] Solved board:")
    print_board(solved_board)

    # 5. Overlay & output
    solved_warped = draw_solution_on_warped(warped, original_board, solved_board)
    result = unwarp_solution(solved_warped, image, M)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"[INFO] Saved result → {output_path}")
    else:
        cv2.imshow("Sudoku Solver — press any key to exit", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OpenCV + Deep Learning Sudoku Solver")
    parser.add_argument("--image",  "-i", help="Path to input sudoku image")
    parser.add_argument("--output", "-o", help="Path to save solved image")
    parser.add_argument("--model",  "-m", default="models/digit_classifier.h5",
                        help="Path to digit classifier model")
    parser.add_argument("--train", action="store_true",
                        help="Train the digit classifier")
    args = parser.parse_args()

    if args.train:
        from src.digit_recognizer import train
        train(save_path=args.model)
        if not args.image:
            return

    if not args.image:
        parser.print_help()
        sys.exit(0)

    run_pipeline(args.image, args.model, args.output)


if __name__ == "__main__":
    main()
