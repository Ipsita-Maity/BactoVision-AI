"""
BactoVision AI - Synthetic Gram-Stain & Colony Image Generator
Produces realistic high-resolution microscopic smear fields and Petri-dish colony images
for benchmarking, training, and zero-setup testing of segmentation and classification pipelines.
"""

import math
import random
import argparse
import numpy as np
import cv2


def _add_microscope_background(width: int, height: int, is_circular_field: bool = True) -> np.ndarray:
    """
    Synthesizes a realistic optical microscope brightfield illumination background
    with non-uniform illumination (vignetting), subtle sensor noise, and warm slide tint.
    """
    # Base warm brightfield color in BGR (light yellowish-beige / glass slide tint)
    bg_b = 215.0
    bg_g = 228.0
    bg_r = 232.0

    # Create radial gradient for optical condenser falloff (vignetting)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2.0, width / 2.0
    max_radius = math.sqrt(center_x**2 + center_y**2)
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Illumination profile: slightly brighter center (condenser focus) fading to edges
    illumination = 1.0 - 0.22 * (dist_from_center / max_radius) ** 1.8
    illumination = np.clip(illumination, 0.65, 1.05)

    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = bg_b * illumination
    base[:, :, 1] = bg_g * illumination
    base[:, :, 2] = bg_r * illumination

    # Add subtle microscopic optical sensor noise
    noise = np.random.normal(0, 3.5, (height, width, 3)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Optional circular aperture mask simulating eyepiece field of view
    if is_circular_field:
        field_mask = np.zeros((height, width), dtype=np.uint8)
        radius = int(min(width, height) * 0.47)
        cv2.circle(field_mask, (int(center_x), int(center_y)), radius, 255, -1)
        # Soft blur boundary
        field_mask = cv2.GaussianBlur(field_mask, (31, 31), 10)
        
        # Dark rim around field
        dark_surround = np.zeros_like(img)
        dark_surround[:, :] = [18, 20, 22] # dark microscope rim
        alpha = (field_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        img = (img * alpha + dark_surround * (1.0 - alpha)).astype(np.uint8)

    # Add faint out-of-focus background debris/stain artifacts
    num_debris = random.randint(6, 15)
    for _ in range(num_debris):
        dx = random.randint(int(width * 0.1), int(width * 0.9))
        dy = random.randint(int(height * 0.1), int(height * 0.9))
        d_rad = random.randint(4, 18)
        d_color = (
            random.randint(180, 205),
            random.randint(190, 215),
            random.randint(200, 220),
        )
        overlay = img.copy()
        cv2.circle(overlay, (dx, dy), d_rad, d_color, -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    return img


def _draw_coccus(img: np.ndarray, center_x: int, center_y: int, radius: int, color_bgr: tuple, blur_kernel: int = 3):
    """Draws a spherical bacterium (Coccus) with realistic 3D cell wall shading."""
    # Outer cell wall
    cv2.circle(img, (center_x, center_y), radius, color_bgr, -1)
    
    # Internal depth gradient: slightly darker core, lighter rim
    core_color = (
        max(0, int(color_bgr[0] * 0.8)),
        max(0, int(color_bgr[1] * 0.8)),
        max(0, int(color_bgr[2] * 0.8))
    )
    if radius > 3:
        cv2.circle(img, (center_x, center_y), max(1, radius - 2), core_color, -1)


def _draw_bacillus(img: np.ndarray, center_x: int, center_y: int, length: int, width: int, angle_deg: float, color_bgr: tuple):
    """Draws a rod-shaped bacterium (Bacillus) with rounded caps and directional orientation."""
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)
    
    half_len = length / 2.0
    cap_radius = int(width / 2.0)
    
    p1 = (int(center_x - dx * (half_len - cap_radius)), int(center_y - dy * (half_len - cap_radius)))
    p2 = (int(center_x + dx * (half_len - cap_radius)), int(center_y + dy * (half_len - cap_radius)))

    # Main rod body
    cv2.line(img, p1, p2, color_bgr, width)
    # Rounded terminal caps
    cv2.circle(img, p1, cap_radius, color_bgr, -1)
    cv2.circle(img, p2, cap_radius, color_bgr, -1)
    
    # Core shading
    core_color = (
        max(0, int(color_bgr[0] * 0.85)),
        max(0, int(color_bgr[1] * 0.85)),
        max(0, int(color_bgr[2] * 0.85))
    )
    if width > 4:
        cv2.line(img, p1, p2, core_color, max(1, width - 4))


def generate_synthetic_microscope_image(
    width: int = 1024,
    height: int = 1024,
    sample_type: str = "mixed",
    num_cells: int = 65,
    is_circular: bool = True,
    random_seed: int = None
) -> np.ndarray:
    """
    Generates a realistic synthetic Gram-stained microscopic view.
    
    Parameters:
    - sample_type: 'mixed', 'gram_positive', 'gram_negative', or 'petri_dish'
    - num_cells: approximate total count of bacterial entities/clusters
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if sample_type == "petri_dish":
        return _generate_petri_dish_image(width, height, num_cells)

    # 1. Base Brightfield Microscope Background
    img = _add_microscope_background(width, height, is_circular_field=is_circular)

    # Valid spawn area (inside circular field if enabled)
    center_x, center_y = width // 2, height // 2
    spawn_radius = int(min(width, height) * 0.38) if is_circular else int(min(width, height) * 0.44)

    # Gram-Positive Colors: Crystal Violet / Deep Indigo-Purple in BGR
    # (Blue, Green, Red) -> high Blue, low Green, medium Red
    pos_colors = [
        (135, 30, 85),
        (155, 35, 95),
        (165, 45, 110),
        (120, 25, 75),
        (140, 30, 90)
    ]

    # Gram-Negative Colors: Safranin / Magenta-Pink-Red in BGR
    # (Blue, Green, Red) -> medium Blue/Pink, low Green, high Red
    neg_colors = [
        (95, 45, 215),
        (110, 50, 230),
        (125, 60, 240),
        (80, 35, 195),
        (105, 50, 220)
    ]

    # 2. Spawn Bacterial Cells
    spawned_count = 0
    while spawned_count < num_cells:
        # Pick random location within field
        r = spawn_radius * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        cx = int(center_x + r * math.cos(theta))
        cy = int(center_y + r * math.sin(theta))

        # Decide Gram type
        if sample_type == "gram_positive":
            is_pos = True
        elif sample_type == "gram_negative":
            is_pos = False
        else: # mixed
            is_pos = random.random() < 0.52

        color = random.choice(pos_colors if is_pos else neg_colors)

        # Decide morphology:
        # Gram positive typically contains Cocci (Staph clusters, Strep chains) or Bacilli (e.g. Bacillus subtilis)
        # Gram negative typically contains Bacilli (E. coli, Pseudomonas) or Diplococci (Neisseria)
        if is_pos:
            morphology_choice = random.choices(["cocci_cluster", "cocci_chain", "bacilli"], weights=[0.60, 0.25, 0.15])[0]
        else:
            morphology_choice = random.choices(["bacilli", "diplococci", "bacilli_cluster"], weights=[0.70, 0.18, 0.12])[0]

        if morphology_choice == "cocci_cluster":
            # Grape-like cluster (Staphylococcus aureus)
            cluster_size = random.randint(4, 12)
            c_rad = random.randint(6, 10)
            for _ in range(cluster_size):
                ox = cx + int(random.gauss(0, c_rad * 1.5))
                oy = cy + int(random.gauss(0, c_rad * 1.5))
                _draw_coccus(img, ox, oy, c_rad, color)
                spawned_count += 1
        elif morphology_choice == "cocci_chain":
            # Linear chain (Streptococcus pneumoniae)
            chain_len = random.randint(3, 7)
            c_rad = random.randint(5, 8)
            ang = random.uniform(0, math.pi)
            for step in range(chain_len):
                ox = int(cx + (step - chain_len / 2.0) * (c_rad * 1.8) * math.cos(ang))
                oy = int(cy + (step - chain_len / 2.0) * (c_rad * 1.8) * math.sin(ang))
                _draw_coccus(img, ox, oy, c_rad, color)
                spawned_count += 1
        elif morphology_choice in ["bacilli", "bacilli_cluster"]:
            # Rods (Escherichia coli or Bacillus subtilis)
            b_count = 1 if morphology_choice == "bacilli" else random.randint(2, 4)
            for _ in range(b_count):
                b_len = random.randint(26, 46)
                b_width = random.randint(9, 14)
                b_angle = random.uniform(0, 360)
                bx = cx + random.randint(-15, 15)
                by = cy + random.randint(-15, 15)
                _draw_bacillus(img, bx, by, b_len, b_width, b_angle, color)
                spawned_count += 1
        elif morphology_choice == "diplococci":
            # Paired cocci (Neisseria)
            c_rad = random.randint(6, 9)
            ang = random.uniform(0, math.pi)
            p1_x = int(cx - (c_rad * 0.9) * math.cos(ang))
            p1_y = int(cy - (c_rad * 0.9) * math.sin(ang))
            p2_x = int(cx + (c_rad * 0.9) * math.cos(ang))
            p2_y = int(cy + (c_rad * 0.9) * math.sin(ang))
            _draw_coccus(img, p1_x, p1_y, c_rad, color)
            _draw_coccus(img, p2_x, p2_y, c_rad, color)
            spawned_count += 2

    # 3. Post-process with realistic optical microscope blur and point-spread function
    img = cv2.GaussianBlur(img, (3, 3), 0.6)

    return img


def _generate_petri_dish_image(width: int = 1024, height: int = 1024, num_colonies: int = 50) -> np.ndarray:
    """
    Generates a realistic agar plate (Petri dish) with bacterial colony forming units (CFUs).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Dark lab bench background
    img[:] = [24, 26, 28]

    center_x, center_y = width // 2, height // 2
    plate_radius = int(min(width, height) * 0.44)

    # Agar medium (translucent yellowish-amber Nutrient Agar)
    agar_color = [185, 215, 230] # BGR
    cv2.circle(img, (center_x, center_y), plate_radius, agar_color, -1)

    # Plastic Petri dish outer rim & meniscus
    cv2.circle(img, (center_x, center_y), plate_radius, (160, 190, 205), 4)
    cv2.circle(img, (center_x, center_y), plate_radius + 6, (70, 75, 80), 3)

    # Colonies: Gram-positive colonies often cream/golden-yellow, Gram-negative often moist pink/translucent
    colony_palettes = [
        # Golden-yellow / creamy Staph
        [(140, 200, 235), (110, 180, 220)],
        # Red/Pink Serratia / Safranin-like
        [(80, 60, 220), (60, 40, 190)],
        # Off-white / pale E. coli
        [(210, 225, 235), (180, 200, 215)],
        # Violet/Deep blue Chromobacterium
        [(170, 50, 90), (140, 30, 70)]
    ]

    for _ in range(num_colonies):
        r = (plate_radius - 25) * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        cx = int(center_x + r * math.cos(theta))
        cy = int(center_y + r * math.sin(theta))

        col_radius = random.randint(8, 26)
        palette = random.choice(colony_palettes)
        rim_color = palette[0]
        center_color = palette[1]

        # Draw smooth circular/slightly undulating colony
        cv2.circle(img, (cx, cy), col_radius, rim_color, -1)
        if col_radius > 6:
            cv2.circle(img, (cx, cy), int(col_radius * 0.65), center_color, -1)

    # Optical soft blur
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic bacterial Gram-stain or Petri dish image.")
    parser.add_argument("--type", type=str, default="mixed", choices=["mixed", "gram_positive", "gram_negative", "petri_dish"],
                        help="Type of sample to generate")
    parser.add_argument("--cells", type=int, default=65, help="Approximate number of cells/colonies")
    parser.add_argument("--out", type=str, default="sample_gram_microscope.jpg", help="Output filepath")
    args = parser.parse_args()

    generated_img = generate_synthetic_microscope_image(
        width=1024, height=1024, sample_type=args.type, num_cells=args.cells
    )
    cv2.imwrite(args.out, generated_img)
    print(f"[OK] Generated {args.type} sample image saved to {args.out}")
