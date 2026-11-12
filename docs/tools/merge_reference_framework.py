#!/usr/bin/env python3
"""Merge the supplied encoder and decoder reference figures."""

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def trim_white(image, margin=8):
    background = Image.new(image.mode, image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox is None:
        return image
    left = max(0, bbox[0] - margin)
    top = max(0, bbox[1] - margin)
    right = min(image.width, bbox[2] + margin)
    bottom = min(image.height, bbox[3] + margin)
    return image.crop((left, top, right, bottom))


def fit_height(image, height):
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def serif_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return font(size, bold=bold)


def arrow(draw, start, end, color=(30, 30, 30), width=7, head=24):
    draw.line((start, end), fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    draw.polygon(
        [(x1, y1), (x1 - head, y1 - head // 2), (x1 - head, y1 + head // 2)],
        fill=color,
    )


def compress_encoder_gap(image, left=600, right=800):
    """Remove the blank middle span while keeping remaining elements unscaled."""
    if not 0 <= left < right <= image.width:
        raise ValueError("invalid encoder compression span")

    shift = right - left
    compressed = Image.new(image.mode, (image.width - shift, image.height), (255, 255, 255))
    compressed.paste(image.crop((0, 0, left, image.height)), (0, 0))
    compressed.paste(image.crop((right, 0, image.width, image.height)), (left, 0))

    # Clear the line fragments at the new seam and reconnect the
    # interaction-context path to the embedding grid.
    draw = ImageDraw.Draw(compressed)
    draw.rectangle((left, 105, 664, 260), fill=(255, 255, 255))
    ink = (20, 20, 20)
    interaction_arrow_x = 875 - shift
    draw.line((543, 125, left, 125), fill=ink, width=4)
    draw.line((left, 125, left, 249), fill=ink, width=4)
    draw.line((left, 249, interaction_arrow_x, 249), fill=ink, width=4)
    draw.polygon(
        [(interaction_arrow_x, 249), (interaction_arrow_x - 13, 241),
         (interaction_arrow_x - 13, 257)],
        fill=ink,
    )
    return compressed, shift


def prepare_decoder(image):
    """Remove decoder query illustrations and route inputs directly to CrossAttn."""
    decoder = image.crop((278, 18, image.width - 6, 635)).copy()
    panel_background = (242, 247, 252)
    draw = ImageDraw.Draw(decoder)

    # Clear the left query illustrations and their labels.
    draw.rectangle((0, 90, 260, decoder.height - 1), fill=panel_background)

    # Direct inputs to the two CrossAttn blocks and preserve their containers.
    arrow(draw, (35, 226), (281, 226), width=3, head=14)
    arrow(draw, (35, 475), (281, 475), width=3, head=14)
    return decoder


def prepare_encoder(image):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    white = (255, 255, 255)
    teal = (46, 154, 168)
    teal_stroke = (30, 107, 118)

    # Remove only the modules that do not exist in the current model path.
    # Keep each mask within the component being replaced.
    draw.rectangle((580, 55, 810, 368), fill=white)       # Transformer Encoder
    draw.rectangle((508, 225, 610, 270), fill=white)      # positional branch
    draw.rectangle((545, 105, 650, 155), fill=white)      # unused attention block
    draw.rectangle((545, 150, 580, 190), fill=white)      # positional arrow
    draw.rectangle((515, 165, 600, 230), fill=white)      # positional annotation
    draw.rectangle((790, 225, 865, 270), fill=white)      # encoder output
    draw.rectangle((970, 99, 1115, 205), fill=white)      # temporal block
    draw.rectangle((1315, 20, 1650, 176), fill=white)     # encoder legend
    draw.rectangle((1290, 176, image.width, image.height), fill=white)
    draw.line((1264, 249, 1345, 249), fill=white, width=9)  # unused branch

    # Connect the interaction-context path directly to the embedding grid.
    ink = (20, 20, 20)
    draw.line((543, 125, 650, 125), fill=ink, width=4)
    draw.line((650, 125, 650, 249), fill=ink, width=4)
    draw.line((650, 249, 875, 249), fill=ink, width=4)
    draw.polygon([(875, 249), (862, 241), (862, 257)], fill=ink)

    # Add state-strength modulation above the history-state grid.
    draw.line((1042, 98, 1042, 140), fill=ink, width=4)
    draw.polygon([(1042, 140), (1034, 128), (1050, 128)], fill=ink)
    draw.rounded_rectangle((974, 140, 1111, 198), radius=10,
                           fill=teal, outline=teal_stroke, width=3)
    draw.text((1042, 169), "State-strength modulation",
              fill=(255, 255, 255), font=serif_font(14, bold=True), anchor="mm")
    draw.line((1042, 198, 1042, 257), fill=ink, width=4)
    draw.polygon([(1042, 257), (1034, 245), (1050, 245)], fill=ink)
    return image


def build(encoder_path, decoder_path, output_path, pdf_path):
    encoder_source = Image.open(encoder_path).convert("RGB")
    encoder_prepared, compression_shift = compress_encoder_gap(
        prepare_encoder(encoder_source)
    )
    encoder_prepared = encoder_prepared.crop(
        (0, 0, min(1650, encoder_prepared.width), encoder_source.height)
    )
    encoder_bbox = ImageChops.difference(
        encoder_prepared, Image.new(encoder_prepared.mode, encoder_prepared.size, (255, 255, 255))
    ).getbbox()
    if encoder_bbox is None:
        raise ValueError("encoder image is empty after preparation")
    trim_left = max(0, encoder_bbox[0] - 8)
    trim_top = max(0, encoder_bbox[1] - 8)
    trim_right = min(encoder_prepared.width, encoder_bbox[2] + 8)
    trim_bottom = min(encoder_prepared.height, encoder_bbox[3] + 8)
    encoder = encoder_prepared.crop((trim_left, trim_top, trim_right, trim_bottom))
    decoder_source = Image.open(decoder_path).convert("RGB")
    decoder = trim_white(prepare_decoder(decoder_source))

    panel_height = 600
    encoder_scale = panel_height / encoder.height
    encoder = fit_height(encoder, panel_height)
    decoder = fit_height(decoder, panel_height)
    gap = 30
    top = 35
    bottom = 35
    canvas_width = encoder.width + gap + decoder.width
    canvas_height = top + panel_height + bottom
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(encoder, (0, top))
    canvas.paste(decoder, (encoder.width + gap, top))

    draw = ImageDraw.Draw(canvas)
    arrow_y = top + round((249 - trim_top) * encoder_scale)
    arrow_start = (
        round((1270 - compression_shift - trim_left) * encoder_scale),
        arrow_y,
    )
    arrow_end = (encoder.width + gap - 15, arrow_y)
    arrow(draw, arrow_start, arrow_end, width=4, head=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(300, 300))
    canvas.save(pdf_path, "PDF", resolution=300.0)
    print(f"wrote {output_path}")
    print(f"wrote {pdf_path}")
    print(f"size {canvas.size[0]}x{canvas.size[1]}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    build(args.encoder, args.decoder, args.output, args.pdf)


if __name__ == "__main__":
    main()
