from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps


ImageInput = Union[str, bytes, np.ndarray]


@dataclass
class EnhancementConfig:
    """
    Configuration class for the enhancement pipeline.
    Change defaults here if you want a different standard pipeline.
    """
    resize_width: Optional[int] = 512
    keep_aspect_ratio: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_denoise: bool = True
    apply_sharpen: bool = True
    sharpen_strength: float = 0.6
    apply_gamma: bool = False
    gamma: float = 1.0
    apply_brightness_contrast: bool = False
    brightness: int = 0
    contrast: int = 0


class ImageEnhancementError(Exception):
    """Custom exception for enhancement-related errors."""
    pass


def _validate_uint8_image(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is a valid uint8 numpy array.

    Args:
        image: Input image array.

    Returns:
        Validated uint8 image.

    Raises:
        ImageEnhancementError: If image is invalid.
    """
    if image is None:
        raise ImageEnhancementError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise ImageEnhancementError("Input image must be a numpy array.")

    if image.size == 0:
        raise ImageEnhancementError("Input image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim not in (2, 3):
        raise ImageEnhancementError(
            f"Unsupported image shape: {image.shape}. Expected 2D or 3D image."
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ImageEnhancementError(
            f"Unsupported channel count: {image.shape[2]}. Expected 1, 3, or 4 channels."
        )

    return image


def _fix_exif_orientation_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes using PIL and correct EXIF orientation.

    Returns:
        BGR uint8 OpenCV image.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = ImageOps.exif_transpose(pil_image)

        if pil_image.mode not in ("RGB", "RGBA", "L"):
            pil_image = pil_image.convert("RGB")

        image_np = np.array(pil_image)

        if image_np.ndim == 2:
            return _validate_uint8_image(image_np)

        if image_np.shape[2] == 4:
            return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)

        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    except Exception as exc:
        raise ImageEnhancementError(f"Failed to load image with EXIF correction: {exc}") from exc


def load_image(image_input: ImageInput, color_mode: str = "color") -> np.ndarray:
    """
    Load an image safely from:
    - file path
    - raw bytes (useful for Streamlit uploaded files)
    - numpy array

    Args:
        image_input: Path, bytes, or numpy array.
        color_mode: 'color', 'grayscale', or 'unchanged'

    Returns:
        Loaded image as numpy array.

    Raises:
        ImageEnhancementError: If loading fails.
    """
    valid_modes = {"color", "grayscale", "unchanged"}
    if color_mode not in valid_modes:
        raise ImageEnhancementError(
            f"Invalid color_mode='{color_mode}'. Use one of {valid_modes}."
        )

    # Case 1: file path
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise ImageEnhancementError(f"Image path does not exist: {image_input}")

        with open(image_input, "rb") as f:
            image_bytes = f.read()

        image = _fix_exif_orientation_from_bytes(image_bytes)

        if color_mode == "grayscale":
            image = convert_to_grayscale(image)
        elif color_mode == "unchanged":
            pass
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return _validate_uint8_image(image)

    # Case 2: bytes (Streamlit uploaded file)
    if isinstance(image_input, bytes):
        image = _fix_exif_orientation_from_bytes(image_input)

        if color_mode == "grayscale":
            image = convert_to_grayscale(image)
        elif color_mode == "unchanged":
            pass
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return _validate_uint8_image(image)

    # Case 3: numpy array
    if isinstance(image_input, np.ndarray):
        image = image_input.copy()
        return _validate_uint8_image(image)

    raise ImageEnhancementError(
        "Unsupported input type. Use image path, bytes, or numpy array."
    )


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB for visualization.

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


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Input image.

    Returns:
        Grayscale image.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
        width: Desired width.
        height: Desired height.
        keep_aspect_ratio: Preserve aspect ratio if only one dimension provided.
        interpolation: OpenCV interpolation method.

    Returns:
        Resized image.
    """
    image = _validate_uint8_image(image)

    if width is None and height is None:
        return image.copy()

    h, w = image.shape[:2]

    if keep_aspect_ratio:
        if width is not None and height is None:
            ratio = width / float(w)
            height = max(1, int(h * ratio))
        elif height is not None and width is None:
            ratio = height / float(h)
            width = max(1, int(w * ratio))
    else:
        if width is None:
            width = w
        if height is None:
            height = h

    if width is None or height is None:
        raise ImageEnhancementError("Width and height could not be determined for resizing.")

    if width <= 0 or height <= 0:
        raise ImageEnhancementError("Width and height must be positive integers.")

    return cv2.resize(image, (width, height), interpolation=interpolation)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization.

    For color images, equalization is applied on the Y channel in YCrCb space.

    Args:
        image: Input image.

    Returns:
        Enhanced image.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return cv2.equalizeHist(image)

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Better than basic histogram equalization for many real images.

    Args:
        image: Input image.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Enhanced image.
    """
    image = _validate_uint8_image(image)

    if clip_limit <= 0:
        raise ImageEnhancementError("clip_limit must be > 0.")

    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or tile_grid_size[0] <= 0
        or tile_grid_size[1] <= 0
    ):
        raise ImageEnhancementError("tile_grid_size must be a tuple of positive integers.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
) -> np.ndarray:
    """
    Apply Gaussian blur.

    Args:
        image: Input image.
        kernel_size: Must be odd numbers.
        sigma_x: Gaussian sigma value.

    Returns:
        Blurred image.
    """
    image = _validate_uint8_image(image)

    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ImageEnhancementError("Gaussian kernel size must contain odd values.")

    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def apply_median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median blur.

    Useful for salt-and-pepper noise.

    Args:
        image: Input image.
        kernel_size: Must be odd and > 1.

    Returns:
        Blurred image.
    """
    image = _validate_uint8_image(image)

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ImageEnhancementError("Median blur kernel size must be an odd integer > 1.")

    return cv2.medianBlur(image, kernel_size)


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp masking.

    Args:
        image: Input image.
        strength: Sharpening intensity. Recommended range: 0.3 to 1.5

    Returns:
        Sharpened image.
    """
    image = _validate_uint8_image(image)

    if strength < 0:
        raise ImageEnhancementError("Sharpen strength cannot be negative.")

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: int = 0,
) -> np.ndarray:
    """
    Adjust brightness and contrast.

    Args:
        image: Input image.
        brightness: Range usually between -100 and 100.
        contrast: Range usually between -100 and 100.

    Returns:
        Adjusted image.
    """
    image = _validate_uint8_image(image)

    alpha = 1.0 + (contrast / 100.0)
    beta = brightness

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction.

    Args:
        image: Input image.
        gamma: Gamma value.
               gamma < 1 makes image brighter,
               gamma > 1 makes image darker.

    Returns:
        Gamma corrected image.
    """
    image = _validate_uint8_image(image)

    if gamma <= 0:
        raise ImageEnhancementError("Gamma must be greater than 0.")

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def denoise_image(
    image: np.ndarray,
    h: int = 10,
    h_color: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    Apply non-local means denoising.

    Args:
        image: Input image.
        h: Filter strength for luminance.
        h_color: Filter strength for color components.
        template_window_size: Template patch size.
        search_window_size: Search window size.

    Returns:
        Denoised image.
    """
    image = _validate_uint8_image(image)

    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h,
            template_window_size,
            search_window_size,
        )

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h,
        h_color,
        template_window_size,
        search_window_size,
    )


def enhance_edges(image: np.ndarray) -> np.ndarray:
    """
    Apply edge enhancement using Laplacian-based sharpening.

    Args:
        image: Input image.

    Returns:
        Edge-enhanced image.
    """
    image = _validate_uint8_image(image)

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    image_float = image.astype(np.float32)
    enhanced = image_float - 0.5 * laplacian.astype(np.float32)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def get_histogram_data(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return histogram data for plotting in Streamlit or Matplotlib.

    For grayscale image: returns one histogram.
    For color image: returns B, G, R histograms.

    Args:
        image: Input image.

    Returns:
        Dictionary of histograms.
    """
    image = _validate_uint8_image(image)
    hist_data: Dict[str, np.ndarray] = {}

    if image.ndim == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_data["gray"] = hist
        return hist_data

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    channels = ("blue", "green", "red")
    for idx, name in enumerate(channels):
        hist = cv2.calcHist([image], [idx], None, [256], [0, 256]).flatten()
        hist_data[name] = hist

    return hist_data


def save_image(output_path: str, image: np.ndarray) -> None:
    """
    Save image safely to disk.

    Args:
        output_path: File path where image will be saved.
        image: Image to save.

    Raises:
        ImageEnhancementError: If save fails.
    """
    image = _validate_uint8_image(image)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success = cv2.imwrite(output_path, image)
    if not success:
        raise ImageEnhancementError(f"Failed to save image to: {output_path}")


def full_enhancement_pipeline(
    image: np.ndarray,
    config: Optional[EnhancementConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply a complete enhancement pipeline and return intermediate outputs.

    Pipeline:
    - Resize
    - CLAHE / Histogram equalization
    - Denoise
    - Brightness / Contrast adjustment
    - Gamma correction
    - Sharpening

    Args:
        image: Input image.
        config: EnhancementConfig object.

    Returns:
        Dictionary containing all intermediate and final images.
    """
    image = _validate_uint8_image(image)
    config = config or EnhancementConfig()

    outputs: Dict[str, np.ndarray] = {}
    outputs["original"] = image.copy()

    # Resize
    resized = resize_image(
        image,
        width=config.resize_width,
        keep_aspect_ratio=config.keep_aspect_ratio,
    )
    outputs["resized"] = resized

    # Contrast enhancement
    if config.apply_clahe:
        contrast_enhanced = apply_clahe(
            resized,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
    else:
        contrast_enhanced = equalize_histogram(resized)

    outputs["contrast_enhanced"] = contrast_enhanced

    # Denoise
    if config.apply_denoise:
        denoised = denoise_image(contrast_enhanced)
    else:
        denoised = contrast_enhanced.copy()

    outputs["denoised"] = denoised

    # Brightness / contrast
    if config.apply_brightness_contrast:
        bc_adjusted = adjust_brightness_contrast(
            denoised,
            brightness=config.brightness,
            contrast=config.contrast,
        )
    else:
        bc_adjusted = denoised.copy()

    outputs["brightness_contrast_adjusted"] = bc_adjusted

    # Gamma correction
    if config.apply_gamma:
        gamma_corrected = apply_gamma_correction(
            bc_adjusted,
            gamma=config.gamma,
        )
    else:
        gamma_corrected = bc_adjusted.copy()

    outputs["gamma_corrected"] = gamma_corrected

    # Sharpen
    if config.apply_sharpen:
        sharpened = sharpen_image(
            gamma_corrected,
            strength=config.sharpen_strength,
        )
    else:
        sharpened = gamma_corrected.copy()

    outputs["sharpened"] = sharpened

    # Edge-enhanced final optional variant
    edge_enhanced = enhance_edges(sharpened)
    outputs["edge_enhanced"] = edge_enhanced

    # Final selected output
    outputs["final"] = sharpened

    return outputs


def process_uploaded_image(uploaded_file: Any) -> np.ndarray:
    """
    Helper for Streamlit uploaded files.

    Example:
        uploaded_file = st.file_uploader(...)
        if uploaded_file is not None:
            image = process_uploaded_image(uploaded_file)

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Loaded OpenCV image (BGR).
    """
    if uploaded_file is None:
        raise ImageEnhancementError("No uploaded file provided.")

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ImageEnhancementError("Uploaded file is empty.")

    return load_image(file_bytes, color_mode="color")


if __name__ == "__main__":
    # Example local testing block
    sample_path = "data/sample_images/sample.jpg"

    try:
        img = load_image(sample_path)
        results = full_enhancement_pipeline(img)

        save_image("data/outputs/original.jpg", results["original"])
        save_image("data/outputs/resized.jpg", results["resized"])
        save_image("data/outputs/contrast_enhanced.jpg", results["contrast_enhanced"])
        save_image("data/outputs/denoised.jpg", results["denoised"])
        save_image(
            "data/outputs/brightness_contrast_adjusted.jpg",
            results["brightness_contrast_adjusted"],
        )
        save_image("data/outputs/gamma_corrected.jpg", results["gamma_corrected"])
        save_image("data/outputs/sharpened.jpg", results["sharpened"])
        save_image("data/outputs/edge_enhanced.jpg", results["edge_enhanced"])
        save_image("data/outputs/final.jpg", results["final"])

        print("Image enhancement pipeline completed successfully.")

    except Exception as exc:
        print(f"Error: {exc}")