"""Page-frame print-instance code shared by the generator and image observer.

The code identifies the printed *file*, not an individual copy made from it.
Its checker border and CRC make an unreadable code a refusal, never an ID guess.
"""

import math
import re
import zlib

SCHEME = 'tatbot.stencil-instance-grid/1'
ROWS, COLS = 6, 34
NONCE_BYTES = 12
_ID = re.compile(r'[0-9a-f]{24}\Z')


def valid_id(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def validate_generation(args):
    if args.instance_id is not None and not valid_id(args.instance_id):
        raise ValueError('instance-id must be 24 lowercase hex digits')
    if args.unmarked and args.instance_id is not None:
        raise ValueError('instance-id requires a printed instance mark')
    if not args.unmarked:
        spec(args.width_mm, args.margin_mm, args.frame_mm)


def spec(width_mm, margin_mm, frame_mm):
    module = min(2., (width_mm-2*margin_mm-4)/COLS, (frame_mm-2)/ROWS)
    if module < 1.2:
        raise ValueError('page top frame is too small for a readable print-instance mark')
    return {'scheme': SCHEME, 'x_mm': round((width_mm-COLS*module)/2, 6),
            'y_mm': round(margin_mm+(frame_mm-ROWS*module)/2, 6),
            'module_mm': round(module, 6), 'rows': ROWS, 'cols': COLS}


def validate(mark, page_mm):
    if not isinstance(mark, dict) or set(mark) != {'scheme', 'x_mm', 'y_mm', 'module_mm', 'rows', 'cols'}:
        raise ValueError('invalid print-instance mark geometry')
    if mark['scheme'] != SCHEME or mark['rows'] != ROWS or mark['cols'] != COLS:
        raise ValueError('unsupported print-instance mark')
    if (not isinstance(page_mm, list) or len(page_mm) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in page_mm)):
        raise ValueError('invalid print-instance page dimensions')
    x, y, module = (mark[key] for key in ('x_mm', 'y_mm', 'module_mm'))
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 500
               for v in (x, y, module)) or module < 1.2:
        raise ValueError('invalid print-instance mark dimensions')
    if x + COLS*module > page_mm[0] or y + ROWS*module > page_mm[1]:
        raise ValueError('print-instance mark extends beyond page')


def data_bits(instance_id):
    if not valid_id(instance_id):
        raise ValueError('print-instance ID must be 24 lowercase hex characters')
    raw = bytes.fromhex(instance_id)
    check = zlib.crc32(b'tatbot-stencil-instance-v1\0' + raw).to_bytes(4, 'big')
    return [(byte >> bit) & 1 for byte in raw + check for bit in range(7, -1, -1)]


def modules(instance_id):
    bits = data_bits(instance_id)
    for row in range(ROWS):
        for col in range(COLS):
            border = row in (0, ROWS-1) or col in (0, COLS-1)
            black = (row+col) % 2 == 0 if border else bool(bits[(row-1)*32+col-1])
            yield row, col, black


def svg_elements(mark, instance_id):
    x, y, unit = mark['x_mm'], mark['y_mm'], mark['module_mm']
    x2, y2 = x+COLS*unit, y+ROWS*unit
    elements = [f'<rect x="{x:.6f}" y="{y:.6f}" width="{x2-x:.6f}" '
                f'height="{y2-y:.6f}" fill="white"/>']
    for row, col, black in modules(instance_id):
        if black:
            left, top = x+col*unit, y+row*unit
            elements.append(f'<rect x="{left:.6f}" y="{top:.6f}" '
                            f'width="{unit:.6f}" height="{unit:.6f}" fill="black"/>')
    return elements


def decode(image, homography, mark):
    """Decode from the current image and current page homography only.

    Returns (nonce, reason); a CRC-valid different nonce is distinguishable
    from an unreadable/occluded mark. No prior frame can supply missing cells.
    """
    import cv2
    import numpy as np

    height, width = image.shape
    x, y, module = (mark[key] for key in ('x_mm', 'y_mm', 'module_mm'))
    page_width, page_height = mark['page_mm']
    samples = np.array([[(x+(col+offset_x)*module)/page_width,
                         (y+(row+offset_y)*module)/page_height]
                        for row in range(ROWS) for col in range(COLS)
                        for offset_x, offset_y in ((.5, .5), (.35, .35), (.65, .35),
                                                   (.35, .65), (.65, .65))], np.float32)
    centers = cv2.perspectiveTransform(samples[None], np.asarray(homography, float))[0]
    if not np.isfinite(centers).all() or (centers[:, 0].min() < 1 or centers[:, 1].min() < 1
            or centers[:, 0].max() >= width-1 or centers[:, 1].max() >= height-1):
        return None, 'instance_mark_out_of_view'
    grid = centers.reshape(ROWS, COLS, 5, 2)
    if (np.min(np.linalg.norm(grid[:, 1:, 0]-grid[:, :-1, 0], axis=2)) < 2.5
            or np.min(np.linalg.norm(grid[1:, :, 0]-grid[:-1, :, 0], axis=2)) < 2.5):
        return None, 'instance_mark_too_small'
    values = cv2.remap(image, centers[:, 0].reshape(-1, 1), centers[:, 1].reshape(-1, 1),
                       cv2.INTER_LINEAR).reshape(ROWS, COLS, 5)
    values = np.median(values, axis=2)
    border = np.fromiter((row in (0, ROWS-1) or col in (0, COLS-1)
                          for row in range(ROWS) for col in range(COLS)), bool).reshape(ROWS, COLS)
    expected_black = np.fromiter(((row+col) % 2 == 0
                                  for row in range(ROWS) for col in range(COLS)), bool).reshape(ROWS, COLS)
    black = float(np.median(values[border & expected_black]))
    white = float(np.median(values[border & ~expected_black]))
    contrast = white-black
    if contrast < 50:
        return None, 'instance_mark_low_contrast'
    threshold = (black+white)/2
    if (np.mean((values[border] < threshold) == expected_black[border]) < .94
            or np.any(np.abs(values[~border]-threshold) < .12*contrast)):
        return None, 'instance_mark_ambiguous'
    payload = (values[1:-1, 1:-1] < threshold).reshape(-1)
    raw = bytes(sum(int(payload[byte*8+bit]) << (7-bit) for bit in range(8))
                for byte in range(16))
    instance_id = raw[:NONCE_BYTES].hex()
    check = zlib.crc32(b'tatbot-stencil-instance-v1\0' + raw[:NONCE_BYTES]).to_bytes(4, 'big')
    if raw[NONCE_BYTES:] != check:
        return None, 'instance_mark_crc_mismatch'
    return instance_id, 'instance_mark_decoded'
