import numpy as np
from PIL import Image
import io
from typing import Union, Tuple, List
from docimind.config import MAX_IMAGE_DIMENSION

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pypdfium2  # type: ignore
    HAS_PYPDFIUM2 = True
except ImportError:
    HAS_PYPDFIUM2 = False

try:
    import pypdf  # type: ignore
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class ImagePreprocessor:
    """
    OpenCV & PyPDF/pypdfium2 based Image Preprocessing module to enhance document quality
    and support both Image files and PDF documents prior to OCR text extraction.
    """

    def __init__(self, max_dim: int = MAX_IMAGE_DIMENSION):
        self.max_dim = max_dim

    @staticmethod
    def _render_text_to_image(text: str, width: int = 1000, height: int = 1400) -> Image.Image:
        """Renders raw text strings onto a clean image canvas for OCR / NLP fallback."""
        from PIL import ImageDraw, ImageFont
        canvas = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        margin = 40
        y = 40
        max_width = width - (2 * margin)

        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                y += 18
                continue
            words = line_str.split()
            curr_line = ""
            for word in words:
                test_line = f"{curr_line} {word}".strip()
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] <= max_width:
                    curr_line = test_line
                else:
                    draw.text((margin, y), curr_line, fill=(0, 0, 0), font=font)
                    y += 22
                    curr_line = word
                    if y > height - 40:
                        break
            if curr_line and y <= height - 40:
                draw.text((margin, y), curr_line, fill=(0, 0, 0), font=font)
                y += 22
            if y > height - 40:
                break
        return canvas

    @staticmethod
    def combine_images_vertically(images: List[Image.Image], max_pages: int = 3) -> Image.Image:
        """Stitches multiple page images vertically into a single continuous canvas."""
        if not images:
            return Image.new('RGB', (1000, 1400), color=(255, 255, 255))
        imgs_to_combine = images[:max_pages]
        if len(imgs_to_combine) == 1:
            return imgs_to_combine[0]

        max_w = max(img.width for img in imgs_to_combine)
        resized_imgs = []
        total_h = 0
        for img in imgs_to_combine:
            w, h = img.size
            if w != max_w and w > 0:
                new_h = int(h * (max_w / w))
                img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
            resized_imgs.append(img)
            total_h += img.height

        combined = Image.new('RGB', (max_w, total_h), color=(255, 255, 255))
        y_offset = 0
        for img in resized_imgs:
            combined.paste(img, (0, y_offset))
            y_offset += img.height

        return combined

    @staticmethod
    def extract_images_from_pdf(pdf_bytes: bytes) -> List[Image.Image]:
        """
        Extracts or renders PIL Images from uploaded PDF bytes.
        Supports pypdfium2 (vector rendering), pypdf page images, and text canvas fallback.
        """
        images = []

        # Priority 1: High-fidelity PDF rendering via pypdfium2 on solid white background
        if HAS_PYPDFIUM2:
            try:
                try:
                    pdf = pypdfium2.PdfDocument(pdf_bytes)
                except Exception:
                    pdf = pypdfium2.PdfDocument(io.BytesIO(pdf_bytes))
                for i in range(len(pdf)):
                    page = pdf[i]
                    try:
                        image = page.render(scale=2.08, fill_color=(255, 255, 255, 255)).to_pil()
                    except Exception:
                        image = page.render(scale=2.08).to_pil()
                    images.append(image)
                if images:
                    return images
            except Exception as e:
                print(f"Warning: Could not render PDF with pypdfium2 ({e}).")


        # Priority 2: Extract embedded raster images via pypdf
        pdf_text = ""
        if HAS_PYPDF:
            try:
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"
                    if hasattr(page, 'images'):
                        for img in page.images:
                            try:
                                images.append(Image.open(io.BytesIO(img.data)))
                            except Exception:
                                pass
            except Exception as e:
                print(f"Warning: Could not extract PDF elements via pypdf ({e}).")

        if images:
            return images

        # Priority 3: Render extracted PDF text onto synthetic image canvas
        if pdf_text.strip():
            rendered_img = ImagePreprocessor._render_text_to_image(pdf_text)
            images.append(rendered_img)
            return images

        # Priority 4: Blank placeholder canvas fallback
        canvas = Image.new('RGB', (1000, 1400), color=(255, 255, 255))
        images.append(canvas)
        return images

    @staticmethod
    def load_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Loads an image or PDF from various input types and converts it to OpenCV BGR numpy array.
        Composites transparent RGBA channels onto solid white background to avoid black background artifacts.
        """
        if isinstance(image_input, Image.Image):
            if image_input.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", image_input.size, (255, 255, 255))
                if image_input.mode == "P":
                    image_input = image_input.convert("RGBA")
                if "A" in image_input.mode:
                    bg.paste(image_input, mask=image_input.split()[-1])
                else:
                    bg.paste(image_input)
                image_input = bg
            elif image_input.mode != "RGB":
                image_input = image_input.convert("RGB")

            arr = np.array(image_input)
            if HAS_CV2:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr

        elif isinstance(image_input, np.ndarray):
            return image_input.copy()

        if isinstance(image_input, bytes):
            # Flexible PDF detection checking for magic bytes anywhere in header
            if b"%PDF" in image_input[:1024]:
                extracted_imgs = ImagePreprocessor.extract_images_from_pdf(image_input)
                combined_img = ImagePreprocessor.combine_images_vertically(extracted_imgs, max_pages=3)
                return ImagePreprocessor.load_image(combined_img)

            if HAS_CV2:
                nparr = np.frombuffer(image_input, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    return image

            # Fallback to PIL
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                return ImagePreprocessor.load_image(pil_img)
            except Exception:
                raise ValueError("Unable to decode image or PDF from bytes.")

        if isinstance(image_input, str):
            if HAS_CV2:
                image = cv2.imread(image_input)
                if image is not None:
                    return image
            try:
                pil_img = Image.open(image_input)
                return ImagePreprocessor.load_image(pil_img)
            except Exception:
                raise ValueError(f"Unable to load image from path: {image_input}")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")


    def resize_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes image while keeping original aspect ratio if dimensions exceed max_dim.
        """
        h, w = image.shape[:2]
        if max(h, w) <= self.max_dim:
            return image

        if h > w:
            new_h = self.max_dim
            new_w = int(w * (self.max_dim / h))
        else:
            new_w = self.max_dim
            new_h = int(h * (self.max_dim / w))

        if HAS_CV2:
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Converts BGR image to single channel grayscale."""
        if len(image.shape) == 2:
            return image
        if HAS_CV2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(image, axis=2).astype(np.uint8)

    @staticmethod
    def denoise(gray_image: np.ndarray) -> np.ndarray:
        """Applies Bilateral Denoising to reduce scanned document noise."""
        if HAS_CV2:
            try:
                return cv2.bilateralFilter(gray_image, 5, 75, 75)
            except Exception:
                try:
                    return cv2.medianBlur(gray_image, 3)
                except Exception:
                    pass
        return gray_image



    @staticmethod
    def adaptive_threshold(gray_image: np.ndarray) -> np.ndarray:
        """Applies Otsu's binarization for high-contrast text segmentation."""
        if HAS_CV2:
            blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        return gray_image

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Detects document skew angle using minAreaRect on binary text mask
        and rotates image to rectify slight scan skew angles.
        """
        if not HAS_CV2:
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))

            if len(coords) < 100:
                return image

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle

            # Only correct realistic document skew angles (between 0.5 and 15.0 degrees)
            if abs(angle) < 0.5 or abs(angle) > 15.0:
                return image

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return image


    def preprocess_pipeline(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        apply_denoise: bool = True,
        apply_deskew: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs full preprocessing sequence on Image or PDF input.
        """
        raw_bgr = self.load_image(image_input)
        resized_bgr = self.resize_aspect_ratio(raw_bgr)

        if apply_deskew and HAS_CV2:
            deskewed_bgr = self.deskew(resized_bgr)
        else:
            deskewed_bgr = resized_bgr

        gray = self.to_grayscale(deskewed_bgr)

        if apply_denoise and HAS_CV2:
            processed_gray = self.denoise(gray)
        else:
            processed_gray = gray

        return deskewed_bgr, processed_gray
