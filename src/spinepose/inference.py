import argparse
import os
import warnings

import cv2

from spinepose.pose_estimator import SpinePoseEstimator
from spinepose.pose_tracker import PoseTracker
from spinepose._version import __version__


def imshow(img, title="Image"):
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    cv2.imshow(title, img)


def infer_image(input_path, mode="medium", output_path=None):
    model = SpinePoseEstimator(mode)

    img = cv2.imread(input_path)
    keypoints, scores = model(img)

    # Show the keypoints on the image
    vis = model.visualize(img, keypoints, scores)
    imshow(vis, "SpinePose Image Inference")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image if requested
    if output_path is not None:
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, vis_rgb)


def infer_video(input_path, mode="medium", use_emoothing=True, output_path=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    pose_tracker = PoseTracker(
        SpinePoseEstimator,
        mode=mode,
        smoothing=use_emoothing,
        smoothing_freq=fps,
    )

    writer = None
    if output_path is not None:
        try:
            import imageio

            writer = imageio.get_writer(output_path, fps=int(fps))
        except ImportError:
            warnings.warn(
                "Please run `pip install imageio[ffmpeg]` to enable video saving."
                " The video will not be saved.",
                UserWarning,
            )
            writer = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        keypoints, scores = pose_tracker(img)
        vis = pose_tracker.visualize(img, keypoints, scores)

        # Display the result
        imshow(vis, "SpinePose Video Inference")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Save the frame if requested
        if writer is not None:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            writer.append_data(vis_rgb)

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.close()


def exists(filepath):
    """Check if the file exists."""
    return os.path.isfile(filepath)


def is_valid(filepath, formats):
    """Check if the file has a valid format."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in formats


def is_image(filename):
    """Check if the file is an image."""
    img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    return exists(filename) and is_valid(filename, img_exts)


def is_video(filename):
    """Check if the file is a video."""
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    return exists(filename) and is_valid(filename, video_exts)


def main():
    parser = argparse.ArgumentParser(description="SpinePose Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="Print the version and exit."
    )
    group.add_argument(
        "--input_path",
        "-i",
        type=str,
        help="Path to the input image or video",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=None,
        help="Path to save the output image or video",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["xlarge", "large", "medium", "small"],
        default="medium",
        help="Model size. Choose from: xlarge, large, medium, small (default: medium)",
    )
    parser.add_argument(
        "--nosmooth",
        action="store_false",
        help="Disable keypoint smoothing for video inference (default: enabled)",
    )
    args = parser.parse_args()

    if args.version:
        print(f"SpinePose {__version__}")
        return

    # Check if the input path is a valid image or video
    if is_image(args.input_path):
        infer_image(
            args.input_path,
            args.mode,
            output_path=args.output_path,
        )
    elif is_video(args.input_path):
        infer_video(
            args.input_path,
            args.mode,
            use_emoothing=args.nosmooth,
            output_path=args.output_path,
        )


if __name__ == "__main__":
    main()
