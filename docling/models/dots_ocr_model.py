"""DotsOCR model integration for Docling.

DotsOCR is a Vision-Language Model (VLM) based OCR engine for multilingual
document layout parsing and text extraction.
See: https://github.com/rednote-hilab/dots.ocr
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Type
from PIL import Image

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import DotsOcrOptions, OcrOptions
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Default model repository
_DOTS_OCR_REPO = "rednote-hilab/dots.ocr"

# Image constraints (from DotsOCR)
MIN_PIXELS = 28 * 28
MAX_PIXELS = 1280 * 28 * 28


class DotsOcrModel(BaseOcrModel):
    """DotsOCR model for multilingual document text extraction.

    This model uses the DotsOCR Vision-Language Model to extract text
    and layout information from document images. DotsOCR supports multiple
    languages and returns structured markdown output with layout information.

    Device Support:
    - CUDA (NVIDIA GPU): Optimal performance with flash_attention_2
    - MPS (Apple Silicon): Supported via standard transformers
    - CPU: Not recommended (very slow)
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: DotsOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: DotsOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi

        # Device and dtype will be set during model initialization
        self.device: Any = None
        self.dtype: Any = None

        if self.enabled:
            self._init_model(accelerator_options, artifacts_path)

    def _init_model(
        self,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Optional[Path],
    ) -> None:
        """Initialize the DotsOCR model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "DotsOCR requires 'transformers', 'torch', and 'qwen-vl-utils' packages. "
                "Please install them via `pip install docling[dotsocr]` to use this OCR engine."
            )

        # Import DotsOCR utilities
        try:
            from docling.models.utils.dots_ocr_util import (
                fetch_image,
                smart_resize,
                get_image_by_fitz_doc,
                layoutjson2md
            )
            # from dots_ocr.utils.layout_utils import post_process_output
            
            self.fetch_image = fetch_image
            self.smart_resize = smart_resize
            self.get_image_by_fitz_doc = get_image_by_fitz_doc
            self.layoutjson2md = layoutjson2md
            
            _log.info("DotsOCR utilities loaded successfully")
        except ImportError as e:
            raise ImportError(
                f"DotsOCR utilities not found: {e}. "
                "Please install dots_ocr package: pip install dots-ocr"
            )

        # Detect available devices
        has_cuda = torch.backends.cuda.is_built() and torch.cuda.is_available()
        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()

        # Determine device and configuration
        if has_cuda:
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16
            self.attn_implementation = "eager"
            #NOTE: 로컬 노트북에서 설치 불가 
            # attn_implementation = self.options.attn_implementation or "flash_attention_2"
            # _log.info(
            #     f"DotsOCR using CUDA device with bfloat16 precision and {attn_implementation} attention"
            # )
        elif has_mps:
            self.device = torch.device("mps")
            self.dtype = torch.float16
            attn_implementation = "eager"
            _log.info("DotsOCR using MPS device (Apple Silicon) with float16 precision")
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            attn_implementation = "eager"
            _log.warning("DotsOCR using CPU device - this will be very slow.")

        # Determine model path
        model_path = self.options.repo_id or _DOTS_OCR_REPO
        if self.options.local_model_path is not None:
            model_path = self.options.local_model_path
        elif artifacts_path is not None:
            repo_cache_folder = (self.options.repo_id or _DOTS_OCR_REPO).replace("/", "--")
            local_path = artifacts_path / repo_cache_folder
            if local_path.exists():
                model_path = str(local_path)

        _log.info(f"Loading DotsOCR model from: {model_path}")

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                # attn_implementation=attn_implementation,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = self.model.eval()

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )

            # Store process_vision_info function
            self.process_vision_info = process_vision_info

            _log.info("DotsOCR model loaded successfully")

        except Exception as e:
            _log.error(f"Failed to load DotsOCR model: {e}")
            raise

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        if ocr_rect.area() == 0:
                            continue

                        # Get image from docling backend
                        # Use scale=1 to get original resolution
                        origin_image = page._backend.get_page_image(
                            scale=1, cropbox=ocr_rect
                        )

                        # Run DotsOCR inference with proper preprocessing
                        cells = self._run_ocr(origin_image, ocr_rect)
                        all_ocr_cells.extend(cells)

                        del origin_image

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    def _preprocess_image(self, origin_image: Image.Image) -> tuple[Image.Image, int, int]:
        """Apply DotsOCR's preprocessing pipeline to the image.
        
        This replicates the exact preprocessing logic from DotsOCRParser._parse_single_image
        
        Returns:
            tuple: (preprocessed_image, input_width, input_height)
        """
        min_pixels = self.options.min_pixels or MIN_PIXELS
        max_pixels = self.options.max_pixels or MAX_PIXELS
        
        # Validate pixel constraints
        if min_pixels < MIN_PIXELS:
            _log.warning(f"min_pixels {min_pixels} < {MIN_PIXELS}, adjusting to {MIN_PIXELS}")
            min_pixels = MIN_PIXELS
        if max_pixels > MAX_PIXELS:
            _log.warning(f"max_pixels {max_pixels} > {MAX_PIXELS}, adjusting to {MAX_PIXELS}")
            max_pixels = MAX_PIXELS
        
        # Apply fitz preprocessing if enabled
        if self.options.fitz_preprocess:
            _log.debug("Applying fitz preprocessing")
            processed_image = self.get_image_by_fitz_doc(origin_image, target_dpi=self.options.dpi)
            processed_image = self.fetch_image(processed_image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            # Direct fetch_image (resizes to fit min/max pixels constraints)
            processed_image = self.fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        
        # Get input dimensions after smart_resize
        input_height, input_width = self.smart_resize(processed_image.height, processed_image.width)
        
        _log.debug(f"Image preprocessing: original={origin_image.size}, processed={processed_image.size}, input=({input_width}, {input_height})")
        
        return processed_image, input_width, input_height

    def _run_ocr(self, origin_image: Image.Image, ocr_rect: BoundingBox) -> list[TextCell]:
        """Run DotsOCR on an image region with DotsOCR's preprocessing pipeline."""
        try:
            # Apply DotsOCR preprocessing
            processed_image, input_width, input_height = self._preprocess_image(origin_image)
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {"type": "text", "text": self.options.prompt}
                    ]
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision information
            image_inputs, video_inputs = self.process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = inputs.to(self.device)

            # Generate output
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                temperature=self.options.temperature,
                top_p=self.options.top_p,
            )

            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode response
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # # Use DotsOCR's post_process_output for proper parsing
            # cells = self._parse_with_post_processing(
            #     response, 
            #     origin_image, 
            #     processed_image, 
            #     ocr_rect
            # )
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(response)
            return response

        except Exception as e:
            _log.warning(f"DotsOCR inference failed: {e}")
            import traceback
            _log.debug(traceback.format_exc())
            return []

    # def _parse_with_post_processing(
    #     self, 
    #     response: str, 
    #     origin_image: Image.Image,
    #     processed_image: Image.Image,
    #     ocr_rect: BoundingBox
    # ) -> list[TextCell]:
    #     """Parse DotsOCR output using the official post_process_output function."""
    #     try:
    #         min_pixels = self.options.min_pixels or MIN_PIXELS
    #         max_pixels = self.options.max_pixels or MAX_PIXELS
            
    #         # Use DotsOCR's post_process_output
    #         cells_data, filtered = self.post_process_output(
    #             response,
    #             self.options.prompt_mode,
    #             origin_image,
    #             processed_image,
    #             min_pixels=min_pixels,
    #             max_pixels=max_pixels,
    #         )

    #         if filtered:
    #             # JSON parsing failed, cells_data is raw markdown text
    #             _log.warning("DotsOCR output JSON parsing failed, using filtered text")
    #             return self._parse_filtered_text(cells_data, ocr_rect)

    #         # cells_data is a list of dicts with 'bbox', 'category', 'text'
    #         text_cells = []
    #         for idx, cell_data in enumerate(cells_data):
    #             bbox = cell_data.get('bbox', [0, 0, origin_image.width, origin_image.height])
    #             category = cell_data.get('category', 'Text')
    #             text = cell_data.get('text', '')

    #             # Skip empty text (except for Pictures)
    #             if not text and category != 'Picture':
    #                 continue

    #             # bbox format from DotsOCR: [x1, y1, x2, y2] in origin_image coordinates
    #             x1, y1, x2, y2 = bbox
                
    #             # Convert from origin_image coordinates to page coordinates
    #             page_x1 = ocr_rect.l + (x1 / origin_image.width) * (ocr_rect.r - ocr_rect.l)
    #             page_y1 = ocr_rect.t + (y1 / origin_image.height) * (ocr_rect.b - ocr_rect.t)
    #             page_x2 = ocr_rect.l + (x2 / origin_image.width) * (ocr_rect.r - ocr_rect.l)
    #             page_y2 = ocr_rect.t + (y2 / origin_image.height) * (ocr_rect.b - ocr_rect.t)

    #             text_cell = TextCell(
    #                 index=idx,
    #                 text=text,
    #                 orig=text,
    #                 from_ocr=True,
    #                 confidence=1.0,
    #                 rect=BoundingRectangle.from_bounding_box(
    #                     BoundingBox.from_tuple(
    #                         coord=(
    #                             page_x1 / self.scale,
    #                             page_y1 / self.scale,
    #                             page_x2 / self.scale,
    #                             page_y2 / self.scale,
    #                         ),
    #                         origin=CoordOrigin.TOPLEFT,
    #                     )
    #                 ),
    #             )
    #             text_cells.append(text_cell)

    #         _log.debug(f"Extracted {len(text_cells)} text cells from DotsOCR output")
    #         return text_cells

    #     except Exception as e:
    #         _log.warning(f"Post-processing failed: {e}")
    #         import traceback
    #         _log.debug(traceback.format_exc())
    #         return self._parse_filtered_text(response, ocr_rect)

    def _parse_filtered_text(self, text: str, ocr_rect: BoundingBox) -> list[TextCell]:
        """Fallback: Parse text when JSON parsing fails (filtered mode)."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            return []

        region_height = ocr_rect.b - ocr_rect.t
        line_height = region_height / len(lines) if lines else region_height

        cells = []
        for idx, line in enumerate(lines):
            top = ocr_rect.t + (idx * line_height)
            bottom = top + line_height

            cell = TextCell(
                index=idx,
                text=line, 
                orig=line,
                from_ocr=True,
                confidence=1.0,
                rect=BoundingRectangle.from_bounding_box(
                    BoundingBox.from_tuple(
                        coord=(
                            ocr_rect.l / self.scale,
                            top / self.scale,
                            ocr_rect.r / self.scale,
                            bottom / self.scale,
                        ),
                        origin=CoordOrigin.TOPLEFT,
                    )
                ),
            )
            cells.append(cell)

        return cells

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return DotsOcrOptions