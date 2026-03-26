"""
utils/feature_extraction.py

Robust feature extraction utilities for the project:
Satellite Image Enhancement and Land Classification using Computer Vision

Features included:
- Safe image loading support through numpy arrays
- Grayscale conversion
- Canny edge detection
- Sobel edge detection
- Laplacian edge detection
- ORB keypoint detection and descriptor extraction
- SIFT keypoint detection and descriptor extraction (if available)
- BRISK keypoint detection
- HOG feature extraction
- Gabor filter-based texture extraction
- Contour extraction
- Image segmentation preview using thresholding
- Feature matching between two images
- Rich helper outputs for Streamlit visualization

Author: Your Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors."""
    pass


@dataclass
class EdgeConfig:
    low_threshold: int = 100
    high_threshold: int = 200
    aperture_size: int = 3
    l2gradient: bool = False


@dataclass
class ORBConfig:
    nfeatures: int = 500
    scale_factor: float = 1.2
    nlevels: int = 8


@dataclass
class SIFTConfig:
    nfeatures: int = 0
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6


@dataclass
class HOGConfig:
    win_size: Tuple[int, int] = (128, 128)
    block_size: Tuple[int, int] = (16, 16)
    block_stride: Tuple[int, int] = (8, 8)
    cell_size: Tuple[int, int] = (8, 8)
    nbins: int = 9


def _validate_uint8_image(image: np.ndarray) -> np.ndarray:
    """
    Validate and normalize image input.

    Args:
        image: Input image.

    Returns:
        Valid uint8 numpy image.
    """
    if image is None:
        raise FeatureExtractionError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise FeatureExtractionError("Input image must be a numpy array.")

    if image.size == 0:
        raise FeatureExtractionError("Input image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim not in (2, 3):
        raise FeatureExtractionError(
            f"Unsupported image dimensions: {image.shape}. Expected 2D or 3D image."
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise FeatureExtractionError(
            f"Unsupported channel count: {image.shape[2]}. Expected 1, 3, or 4 channels."
        )

    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR/RGBA image to grayscale.

    Args:
        image: Input image.

    Returns:
        Grayscale image.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return image

    if image.shape[2] == 1:
        return image[:, :, 0]

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR to RGB for Streamlit/Matplotlib display.

    Args:
        image: Input image.

    Returns:
        RGB image.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize image safely.

    Args:
        image: Input image.
        width: Target width.
        height: Target height.
        keep_aspect_ratio: Whether to keep aspect ratio.
        interpolation: OpenCV interpolation.

    Returns:
        Resized image.
    """
    image = _validate_uint8_image(image)
    h, w = image.shape[:2]

    if width is None and height is None:
        return image.copy()

    if keep_aspect_ratio:
        if width is not None and height is None:
            ratio = width / float(w)
            height = int(h * ratio)
        elif height is not None and width is None:
            ratio = height / float(h)
            width = int(w * ratio)

    if width is None:
        width = w
    if height is None:
        height = h

    if width <= 0 or height <= 0:
        raise FeatureExtractionError("Width and height must be positive.")

    return cv2.resize(image, (width, height), interpolation=interpolation)


def detect_canny_edges(
    image: np.ndarray,
    config: Optional[EdgeConfig] = None,
) -> np.ndarray:
    """
    Detect edges using Canny algorithm.

    Args:
        image: Input image.
        config: Edge configuration.

    Returns:
        Edge map.
    """
    config = config or EdgeConfig()
    gray = convert_to_grayscale(image)

    return cv2.Canny(
        gray,
        threshold1=config.low_threshold,
        threshold2=config.high_threshold,
        apertureSize=config.aperture_size,
        L2gradient=config.l2gradient,
    )


def detect_sobel_edges(image: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
    """
    Detect edges using Sobel operator.

    Args:
        image: Input image.
        ksize: Kernel size.

    Returns:
        Dictionary with x, y, and magnitude gradients.
    """
    gray = convert_to_grayscale(image)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    magnitude_abs = cv2.convertScaleAbs(magnitude)

    return {
        "sobel_x": sobel_x_abs,
        "sobel_y": sobel_y_abs,
        "magnitude": magnitude_abs,
    }


def detect_laplacian_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Detect edges using Laplacian operator.

    Args:
        image: Input image.
        ksize: Kernel size.

    Returns:
        Laplacian edge image.
    """
    gray = convert_to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)


def detect_orb_features(
    image: np.ndarray,
    config: Optional[ORBConfig] = None,
    draw_keypoints: bool = True,
) -> Dict[str, Any]:
    """
    Detect ORB keypoints and descriptors.

    Args:
        image: Input image.
        config: ORB configuration.
        draw_keypoints: Whether to draw keypoints on image.

    Returns:
        Dictionary containing keypoints, descriptors, and visualization image.
    """
    config = config or ORBConfig()
    gray = convert_to_grayscale(image)

    orb = cv2.ORB_create(
        nfeatures=config.nfeatures,
        scaleFactor=config.scale_factor,
        nlevels=config.nlevels,
    )
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    output = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "num_keypoints": len(keypoints),
    }

    if draw_keypoints:
        if gray.ndim == 2:
            draw_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            draw_base = image.copy()

        keypoint_image = cv2.drawKeypoints(
            draw_base,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        output["keypoint_image"] = keypoint_image

    return output


def detect_sift_features(
    image: np.ndarray,
    config: Optional[SIFTConfig] = None,
    draw_keypoints: bool = True,
) -> Dict[str, Any]:
    """
    Detect SIFT keypoints and descriptors.

    Note:
        SIFT may not be available in all OpenCV installations.
        Install opencv-contrib-python if needed.

    Args:
        image: Input image.
        config: SIFT configuration.
        draw_keypoints: Whether to create visualization image.

    Returns:
        Dictionary containing keypoints, descriptors, and image.

    Raises:
        FeatureExtractionError: If SIFT is unavailable.
    """
    config = config or SIFTConfig()
    gray = convert_to_grayscale(image)

    if not hasattr(cv2, "SIFT_create"):
        raise FeatureExtractionError(
            "SIFT is not available in this OpenCV installation. "
            "Install opencv-contrib-python."
        )

    sift = cv2.SIFT_create(
        nfeatures=config.nfeatures,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    output = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "num_keypoints": len(keypoints),
    }

    if draw_keypoints:
        keypoint_image = cv2.drawKeypoints(
            gray,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        output["keypoint_image"] = keypoint_image

    return output


def detect_brisk_features(image: np.ndarray, draw_keypoints: bool = True) -> Dict[str, Any]:
    """
    Detect BRISK keypoints and descriptors.

    Args:
        image: Input image.
        draw_keypoints: Whether to draw keypoints.

    Returns:
        Dictionary of keypoints, descriptors, and visualization.
    """
    gray = convert_to_grayscale(image)

    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(gray, None)

    output = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "num_keypoints": len(keypoints),
    }

    if draw_keypoints:
        keypoint_image = cv2.drawKeypoints(
            gray,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        output["keypoint_image"] = keypoint_image

    return output


def extract_hog_features(
    image: np.ndarray,
    config: Optional[HOGConfig] = None,
) -> Dict[str, Any]:
    """
    Extract HOG (Histogram of Oriented Gradients) features.

    Args:
        image: Input image.
        config: HOG configuration.

    Returns:
        Dictionary with HOG feature vector and processed image.
    """
    config = config or HOGConfig()
    gray = convert_to_grayscale(image)

    resized = resize_image(gray, width=config.win_size[0], height=config.win_size[1], keep_aspect_ratio=False)

    hog = cv2.HOGDescriptor(
        _winSize=config.win_size,
        _blockSize=config.block_size,
        _blockStride=config.block_stride,
        _cellSize=config.cell_size,
        _nbins=config.nbins,
    )

    features = hog.compute(resized)

    return {
        "resized_image": resized,
        "hog_features": features,
        "feature_length": int(features.shape[0]) if features is not None else 0,
    }


def apply_gabor_filters(
    image: np.ndarray,
    kernel_size: int = 21,
    sigma: float = 5.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0,
    orientations: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply Gabor filters for texture analysis.

    Args:
        image: Input image.
        kernel_size: Gabor kernel size.
        sigma: Standard deviation.
        lambd: Wavelength.
        gamma: Aspect ratio.
        psi: Phase offset.
        orientations: List of orientation angles in radians.

    Returns:
        Dictionary of Gabor responses.
    """
    gray = convert_to_grayscale(image)

    if orientations is None:
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    responses: Dict[str, np.ndarray] = {}

    for i, theta in enumerate(orientations):
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            theta,
            lambd,
            gamma,
            psi,
            ktype=cv2.CV_32F,
        )
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        responses[f"orientation_{i}"] = filtered

    return responses


def find_contours(
    image: np.ndarray,
    threshold: int = 127,
    min_area: float = 10.0,
    draw_contours: bool = True,
) -> Dict[str, Any]:
    """
    Find contours in the image.

    Args:
        image: Input image.
        threshold: Threshold value for binarization.
        min_area: Minimum contour area to keep.
        draw_contours: Whether to draw contours on image.

    Returns:
        Dictionary containing contours and visualization.
    """
    gray = convert_to_grayscale(image)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    output = {
        "binary": binary,
        "contours": filtered_contours,
        "num_contours": len(filtered_contours),
    }

    if draw_contours:
        contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
        output["contour_image"] = contour_image

    return output


def segment_image_threshold(image: np.ndarray, threshold: int = 127) -> Dict[str, np.ndarray]:
    """
    Basic threshold-based segmentation.

    Args:
        image: Input image.
        threshold: Binary threshold.

    Returns:
        Dictionary with grayscale and binary segmented image.
    """
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return {
        "gray": gray,
        "segmented": binary,
    }


def match_features_orb(
    image1: np.ndarray,
    image2: np.ndarray,
    max_matches_to_draw: int = 50,
) -> Dict[str, Any]:
    """
    Match ORB features between two images.

    Args:
        image1: First image.
        image2: Second image.
        max_matches_to_draw: Maximum number of matches to display.

    Returns:
        Dictionary with keypoints, matches, and visualization image.
    """
    result1 = detect_orb_features(image1, draw_keypoints=False)
    result2 = detect_orb_features(image2, draw_keypoints=False)

    kp1 = result1["keypoints"]
    kp2 = result2["keypoints"]
    des1 = result1["descriptors"]
    des2 = result2["descriptors"]

    if des1 is None or des2 is None:
        raise FeatureExtractionError("Could not compute ORB descriptors for one or both images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img1_color = cv2.cvtColor(convert_to_grayscale(image1), cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(convert_to_grayscale(image2), cv2.COLOR_GRAY2BGR)

    matched_image = cv2.drawMatches(
        img1_color,
        kp1,
        img2_color,
        kp2,
        matches[:max_matches_to_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return {
        "keypoints1": kp1,
        "keypoints2": kp2,
        "descriptors1": des1,
        "descriptors2": des2,
        "matches": matches,
        "num_matches": len(matches),
        "matched_image": matched_image,
    }


def get_keypoint_coordinates(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """
    Convert keypoints into coordinate array.

    Args:
        keypoints: List of OpenCV keypoints.

    Returns:
        Nx2 array of point coordinates.
    """
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return coords


def full_feature_extraction_pipeline(image: np.ndarray) -> Dict[str, Any]:
    """
    Run a full feature extraction pipeline.

    Includes:
    - Canny edges
    - Sobel edges
    - Laplacian edges
    - ORB keypoints
    - BRISK keypoints
    - HOG features
    - Gabor texture responses
    - Contours
    - Threshold segmentation

    Args:
        image: Input image.

    Returns:
        Dictionary containing all extracted outputs.
    """
    image = _validate_uint8_image(image)

    results: Dict[str, Any] = {}
    results["original"] = image.copy()
    results["gray"] = convert_to_grayscale(image)

    # Edge detection
    results["canny_edges"] = detect_canny_edges(image)
    sobel_data = detect_sobel_edges(image)
    results["sobel_x"] = sobel_data["sobel_x"]
    results["sobel_y"] = sobel_data["sobel_y"]
    results["sobel_magnitude"] = sobel_data["magnitude"]
    results["laplacian_edges"] = detect_laplacian_edges(image)

    # Keypoint features
    orb_data = detect_orb_features(image)
    results["orb_keypoints"] = orb_data["keypoints"]
    results["orb_descriptors"] = orb_data["descriptors"]
    results["orb_num_keypoints"] = orb_data["num_keypoints"]
    results["orb_keypoint_image"] = orb_data["keypoint_image"]

    brisk_data = detect_brisk_features(image)
    results["brisk_keypoints"] = brisk_data["keypoints"]
    results["brisk_descriptors"] = brisk_data["descriptors"]
    results["brisk_num_keypoints"] = brisk_data["num_keypoints"]
    results["brisk_keypoint_image"] = brisk_data["keypoint_image"]

    # SIFT if available
    try:
        sift_data = detect_sift_features(image)
        results["sift_available"] = True
        results["sift_keypoints"] = sift_data["keypoints"]
        results["sift_descriptors"] = sift_data["descriptors"]
        results["sift_num_keypoints"] = sift_data["num_keypoints"]
        results["sift_keypoint_image"] = sift_data["keypoint_image"]
    except Exception:
        results["sift_available"] = False
        results["sift_keypoints"] = None
        results["sift_descriptors"] = None
        results["sift_num_keypoints"] = 0
        results["sift_keypoint_image"] = None

    # HOG
    hog_data = extract_hog_features(image)
    results["hog_resized_image"] = hog_data["resized_image"]
    results["hog_features"] = hog_data["hog_features"]
    results["hog_feature_length"] = hog_data["feature_length"]

    # Gabor
    gabor_data = apply_gabor_filters(image)
    results["gabor_responses"] = gabor_data

    # Contours
    contour_data = find_contours(image)
    results["binary_contours"] = contour_data["binary"]
    results["contours"] = contour_data["contours"]
    results["num_contours"] = contour_data["num_contours"]
    results["contour_image"] = contour_data["contour_image"]

    # Segmentation
    segmentation_data = segment_image_threshold(image)
    results["segmented_image"] = segmentation_data["segmented"]

    return results


if __name__ == "__main__":
    # Example local testing block
    sample_path = "data/sample_images/sample.jpg"

    try:
        image = cv2.imread(sample_path)
        if image is None:
            raise FeatureExtractionError(f"Could not load sample image from {sample_path}")

        outputs = full_feature_extraction_pipeline(image)

        cv2.imwrite("data/outputs/canny_edges.jpg", outputs["canny_edges"])
        cv2.imwrite("data/outputs/sobel_x.jpg", outputs["sobel_x"])
        cv2.imwrite("data/outputs/sobel_y.jpg", outputs["sobel_y"])
        cv2.imwrite("data/outputs/sobel_magnitude.jpg", outputs["sobel_magnitude"])
        cv2.imwrite("data/outputs/laplacian_edges.jpg", outputs["laplacian_edges"])
        cv2.imwrite("data/outputs/orb_keypoints.jpg", outputs["orb_keypoint_image"])
        cv2.imwrite("data/outputs/brisk_keypoints.jpg", outputs["brisk_keypoint_image"])
        cv2.imwrite("data/outputs/contours.jpg", outputs["contour_image"])
        cv2.imwrite("data/outputs/segmented.jpg", outputs["segmented_image"])

        if outputs["sift_available"] and outputs["sift_keypoint_image"] is not None:
            cv2.imwrite("data/outputs/sift_keypoints.jpg", outputs["sift_keypoint_image"])

        for name, gabor_img in outputs["gabor_responses"].items():
            cv2.imwrite(f"data/outputs/{name}.jpg", gabor_img)

        print("Feature extraction pipeline completed successfully.")

    except Exception as exc:
        print(f"Error: {exc}")