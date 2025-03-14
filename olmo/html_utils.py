"""HTML utilities for visualizing datasets, preprocessing, or predictions"""
import base64
import io
import re
from dataclasses import dataclass
from io import BytesIO
from typing import List, Dict, Any, Optional

import PIL.Image
import numpy as np
from einops import einops
from html import escape as html_escape

from olmo import tokenizer
from olmo.util import extract_points
from olmo.tokenizer import get_special_token_ids

COLORS = [
    "aqua",
    "black",
    "blue",
    "fuchsia",
    "gray",
    "green",
    "lime",
    "maroon",
    "navy",
    "olive",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow"
]


def unnormalize_image(image,
                      offset=(0.48145466, 0.4578275, 0.40821073),
                      scale=(0.26862954, 0.26130258, 0.27577711)):
    """Normalizes the image to zero mean and unit variance."""
    image *= np.array(scale)[None, None, :]
    image += np.array(offset)[None, None, :]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def escape_html(text):
    return "<br>".join(html_escape(x) for x in text.split("\n"))


def example_to_html_dict(ex, preprocessor, show_patches=False, show_crops=False):
    """Build HTML visualizations for an examples

    ex: The example (after preprocessing) to show
    preprocessor: The preprocessor used to preprocessor the examples
    show_patches: Whether to visualize the image features as patches
    show_crops: Whether to visualize crops used
    """
    if "metadata" in ex:
        metadata = ex["metadata"]
    else:
        metadata = {k[len("metadata/"):]: v for k, v in
                    ex.items() if k.startswith("metadata/")}
    voc = preprocessor.tokenizer

    boxes = []
    if "subsegment_ids" in ex:
        targets = ex["input_tokens"].ravel()
        subsegment_ids = ex["subsegment_ids"]
        shared_prefix = postprocess_prompt(voc.decode(targets[subsegment_ids == 0]))
        segment_text = []
        for i in np.sort(np.unique(subsegment_ids)):
            if i == -1:
                continue
            mask = subsegment_ids == i
            segment_text.append((i, voc.decode(targets[mask], False), ex["loss_masks"][mask].mean()))

        text = []
        text.append("<ul>")
        text.append(str(ex['images'].shape))
        for i, seg, w in segment_text:
            seg = postprocess_prompt(seg)
            text.append("<li>")
            if "image_size" in ex["metadata"]:
                seg_points = extract_points(seg, *ex["metadata"]["image_size"])
            else:
                seg_points = []
            if seg_points:
                color = COLORS[i % len(COLORS)]
                text.append(f"<span style=\"color: {color}\">SEGMENT {i}</span> w={w:0.3f}: " + escape_html(seg))
                boxes.append(BoxesToVisualize([[x-5, y-5, x+5, y+5] for x, y in seg_points], color, "xyxy"))
            else:
                text.append(f"SEGMENT {i} w={w:0.3f}: " + escape_html(seg))
            text.append("</li>")
        text.append("</ul>")
        text = " ".join(text)
    else:
        text = voc.decode(ex["target_tokens"][ex["target_tokens"] != voc.pad_id])
        if "image_size" in metadata:
            points = extract_points(text, *metadata["image_size"])
            boxes = [BoxesToVisualize([[x-5, y-5, x+5, y+5] for x, y in points], "blue", "xyxy")]
        text = escape_html(postprocess_prompt(text))
    out = dict(text=text)

    image_src = None
    if "image_url" in metadata:
        image_src = metadata["image_url"]
    elif "image" in metadata:
        image = metadata["image"]
        if isinstance(image, bytes):
            image_src = f'data:image/jpeg;base64,{base64.b64encode(image).decode()}'
        elif isinstance(image, str):
            image_src = image
        else:
            image_src = build_embedded_image(image)

    if image_src is not None:
        max_dim = 768
        if len(boxes) == 0:
            out["image"] = f"<img style=\"max-height:{max_dim}px;max-width:{max_dim}px;height:auto;width:auto;\" src={image_src}><img>"
        else:
            out["image"] = get_html_image_with_boxes(
                image_src, boxes,
                img_size=metadata.get("image_size"),
                max_dim=max_dim
            )

    patch_size = preprocessor.mm_preprocessor.image_patch_size
    base_h, base_w = preprocessor.mm_preprocessor.base_image_input_size
    images = einops.rearrange(
        ex["images"], 't (h w) (dh dw c) -> t (h dh) (w dw) c',
        h=base_h//patch_size,
        w=base_w//patch_size,
        dh=patch_size, dw=patch_size, c=3)
    images = unnormalize_image(images)

    if show_crops:
        n_crops = ex["images"].shape[0]
        crop_h, crop_w = base_h//patch_size, base_w//patch_size
        boxes_to_show = [[] for _ in range(len(images))]
        patches_used = []
        if ex.get("pooled_patches_idx") is not None:
            patches_used += ex["pooled_patches_idx"].tolist()
        if ex.get("low_res_pooled_idx") is not None:
            patches_used += ex["low_res_pooled_idx"].tolist()
        if ex.get("high_res_pooled_idx") is not None:
            patches_used += ex["high_res_pooled_idx"].tolist()

        for patch_ids in patches_used:
            patch_ids = np.array(patch_ids)
            patch_ids = patch_ids[patch_ids >= 0]
            if len(patch_ids) == 0:
                continue
            crop_ix = patch_ids.max() // (crop_h * crop_w)
            patch_ids %= (crop_h * crop_w)
            xs = (patch_ids % crop_h) * patch_size
            ys = (patch_ids // crop_h) * patch_size
            boxes_to_show[crop_ix].append([
                xs.min(), ys.min(), xs.max()+patch_size, ys.max()+patch_size])

        for crop_ix, (crop, boxes) in enumerate(zip(images, boxes_to_show)):
            if len(boxes) > 0:
                out[f"patch-{crop_ix}"] = get_html_image_with_boxes(
                    build_embedded_image(crop), [BoxesToVisualize(np.array(boxes), "blue")])
            else:
                out[f"patch-{crop_ix}"] = crop

    if show_patches:
        pooled_patches_idx = ex["pooled_patches_idx"]
        special_token_to_id = get_special_token_ids(voc)
        image_patch_id = special_token_to_id[tokenizer.IMAGE_PATCH_TOKEN]
        id_to_special_token = {i: k for i, k in special_token_to_id.items()}
        with_patches = []
        patches = einops.rearrange(images,
            't (h dh) (w dw) c -> (t h w) dh dw c',
            dh=patch_size, dw=patch_size
        )
        on_pooled_patch = 0
        for token_ix, ix in enumerate(ex["input_tokens"]):
            if ix == -1:
                with_patches.append("<PAD>")
            elif ix == image_patch_id:
                # [pool_h, pool_w, patch_h, patch_w, dim]
                sub_patches = patches[pooled_patches_idx[on_pooled_patch]]
                patch = einops.rearrange(
                    sub_patches,
                    '(pool_h pool_w) patch_h patch_w c -> (pool_h patch_h) (pool_w patch_w) c',
                    pool_h=preprocessor.mm_preprocessor.image_pooling_h,
                    pool_w=preprocessor.mm_preprocessor.image_pooling_w
                )

                src = build_embedded_image(patch)
                with_patches.append(f"<img src={src}></img>")
                on_pooled_patch += 1
            elif ix in id_to_special_token:
                with_patches.append(html_escape(str(id_to_special_token)))
            else:
                with_patches.append(html_escape(voc.decode([ix])))
        out["tokens"] = " ".join(with_patches)
    return out


def build_embedded_image(image_data):
    """Turns an image into a string that can be used as a src in html images"""
    if image_data.dtype == np.float32:
        image_data = (image_data*255).astype(np.uint8)
    with PIL.Image.fromarray(image_data) as img:
        image_data = io.BytesIO()
        img.save(image_data, format='JPEG')
        image_data = image_data.getvalue()
    encoded_image = base64.b64encode(image_data)
    return f'data:image/jpeg;base64,{encoded_image.decode()}'


def build_html_table(data: List[Dict[str, Any]], col_widths=None, fixed_width=False) -> str:
    columns = {}  # Collect any key that appears in the data, in order
    for row in data:
        for key in row:
            columns[key] = None
    html = [
"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta content="text/html;charset=utf-8" http-equiv="Content-Type">
  <meta content="utf-8" http-equiv="encoding">
</head>
""".strip()
    ]

    # Body
    html.append("<body>")
    if fixed_width:
        html.append("<table style=\"table-layout: fixed; width:100%\">")
    else:
        html.append("<table>")

    # Table Header
    html.append("<tr>")
    header = []
    for c in columns:
        if col_widths and c in col_widths:
            header.append(f"<th style=\"{col_widths[c]}\">{c}</th>")
        else:
            header.append(f"<th>{c}</th>")
    html.append(" ".join(header))
    html.append("</tr>")

    # Table Body
    for ex in data:
        cells = []
        for c in columns:
            val = ex.get(c)
            if val is None:
                cells.append("")
            elif isinstance(val, str):
                cells.append(val)
            elif isinstance(val, (float, int)):
                cells.append(val)
            elif len(val.shape) == 3 and val.shape[-1] == 3:
                # Assume an image
                data = build_embedded_image(val)
                cells.append(f'<img src={data}></img>')
            else:
                raise NotImplementedError(f"Data not understood for {val.shape}: {val}")
        html.append("<tr>")
        html.append("\n".join(f"<td>{x}</td>" for x in cells))
        html.append("</tr>")

    html.append("</table>")
    html.append("</body>")
    html.append("</html>")
    return "\n".join(html)


@dataclass
class BoxesToVisualize:
    """Boxes to draw on an image"""
    boxes: Any
    color: str
    format: str = "yxyx"
    labels: List[str] = None
    shape: str = "box"


def html_rect(x1, y1, x2, y2, color="black", border_width="medium", label=None):
    """Utility method to get a HTML rectangle element"""
    rect_style = {
        "position": "absolute",
        "top": f"{y1}px",
        "left": f"{x1}px",
        "height": f"{y2-y1}px",
        "width": f"{x2-x1}px",
        "border-style": "solid",
        "border-color": color,
        "border-width": border_width,
        "box-sizing": "border-box",
    }
    rect_style_str = "; ".join(f"{k}: {v}" for k, v in rect_style.items())

    text_style = {
        "position": "absolute",
        "top": y1-5,
        "left": x1+3,
        "color": color,
        "background-color": "black",
        "z-index": 9999,
        "padding-right": "5px",
        "padding-left": "5px",
    }
    text_style_str = "; ".join(f"{k}: {v}" for k, v in text_style.items())

    if label is None:
        text = ''
    else:
        text = f'  <div style="{text_style_str}">{label}</div>'

    if text:
        html = [f'<span style="{rect_style_str}"></span>']
    else:
        html = [
            f'<div>',
            f'  <div style="{rect_style_str}"></div>',
            text,
            "</div>"
        ]
    return html


def get_html_image_with_boxes(
    image_src, boxes: List[BoxesToVisualize],
    width=None, height=None, wrap="div", img_size=None, max_dim=None) -> str:
    """Build a HTML element containing `image_src` and the boxes in `boxes` on top of it.

    Provides a way to draw annotated images without have to load/modify the image itself
    """
    html = []
    html += [f'<{wrap} style="display: inline-block; position: relative;">']
    image_attr = dict(src=image_src)
    if max_dim is not None:
        assert height is None and width is None
        scale = max_dim / max(img_size)
        if scale > 0:
            width = round(img_size[0]*scale)
            height = round(img_size[1]*scale)

    if width:
        image_attr["width"] = width
    if height:
        image_attr["height"] = height
    attr_str = " ".join(f"{k}={v}" for k, v in image_attr.items())
    html += [f'<img {attr_str}>']

    for box_set in boxes:
        if height or width:
            img_w, img_h = img_size
            if not width:
                factor = height/img_h
                w_factor = factor
                h_factor = factor
            elif height and width:
                w_factor = width/img_w
                h_factor = height/img_h
            else:
                raise NotImplementedError()
        else:
            w_factor = 1
            h_factor = 1

        if boxes is not None and len(boxes) > 0:
            task_boxes = np.asarray(box_set.boxes)
            if box_set.format == "yxyx":
                task_boxes = np.stack([
                    task_boxes[:, 1], task_boxes[:, 0],
                    task_boxes[:, 3], task_boxes[:, 2],
                ], -1)
            elif box_set.format == "xyxy":
                pass
            elif box_set.format == "xywh":
                task_boxes = np.stack([
                    task_boxes[:, 0], task_boxes[:, 1],
                    task_boxes[:, 0] + task_boxes[:, 2],
                    task_boxes[:, 1] + task_boxes[:, 3]
                ], -1)
            else:
                raise NotImplementedError(box_set.format)

        for ix in range(len(task_boxes)):
            box = task_boxes[ix]
            x1, y1, x2, y2 = box
            html += html_rect(
                x1*w_factor, y1*h_factor, x2*w_factor, y2*h_factor,
                color=box_set.color,
                label=None if box_set.labels is None else box_set.labels[ix]
            )

    html += [f'</{wrap}>']
    return "\n".join(html)


def postprocess_prompt(prompt_text):
    """Get a human-readable prompt by compressing the image tokens"""
    start = 0
    prompt_text = prompt_text.lstrip()  # some tokenizers add a leading space before special tokens
    post_processed_text = ""
    for match in re.finditer(r"<im_start>\s?((<im_patch>|<im_col>)\s?)*\s?<im_end>", prompt_text):
        n_patches = match.group(0).count("<im_patch>")
        if match.start() > start:
            post_processed_text += prompt_text[start:match.start()]
        post_processed_text += f"IMAGE[{n_patches}]"
        start = match.end()
    post_processed_text += prompt_text[start:]
    return post_processed_text
