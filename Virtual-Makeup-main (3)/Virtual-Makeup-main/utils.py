import cv2
import numpy as np
from landmarks import detect_landmarks, normalize_landmarks
from config import FOUNDATION_SHADES
from config import app_state

# Comprehensive face contour landmarks for precise coverage
FACE_CONTOUR = [
    # Core face contour - precise points only
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Specific areas that need coverage
RIGHT_CHEEK = [347, 348, 349, 350, 280, 329, 330, 331, 266, 426]
LEFT_CHEEK = [118, 119, 120, 121, 50, 101, 100, 99, 116, 117]
NOSE_SIDES = [129, 98, 97, 2, 326, 327, 294, 278, 209, 198]

# Exclusion areas
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def get_shade_bgr(shade_key):
    """Get BGR values for a specific shade"""
    shade_info = FOUNDATION_SHADES.get(shade_key, FOUNDATION_SHADES['03'])
    return shade_info['bgr']

def detect_landmarks_uploaded_image(src):
    """
    Special landmark detection for uploaded images (static images)
    Uses static_image_mode=True for better accuracy
    """
    print("🔍 Detecting landmarks for uploaded image...")
    from mediapipe.python.solutions.face_mesh import FaceMesh
    
    with FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        print(f"✅ Face detected with {len(results.multi_face_landmarks[0].landmark)} landmarks")
        return results.multi_face_landmarks[0].landmark
    else:
        print("❌ No face detected in uploaded image")
        return None

def create_precise_face_mask(landmarks, height, width):
    """Create a precise mask that stays within face boundaries"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Method 1: Use precise face contour with convex hull
        face_points = normalize_landmarks(landmarks, height, width, FACE_CONTOUR)
        if face_points.size > 0 and len(face_points) > 2:
            face_hull = cv2.convexHull(face_points)
            cv2.fillPoly(mask, [face_hull], 255)
        
        # Method 2: Add cheek areas with precise boundaries
        cheek_mask = create_precise_cheek_mask(landmarks, height, width)
        mask = cv2.bitwise_or(mask, cheek_mask)
        
        # Method 3: Create tight face boundary
        boundary_mask = create_tight_face_boundary(landmarks, height, width)
        mask = cv2.bitwise_and(mask, boundary_mask)
        
        # Create exclusion masks
        exclusion_mask = create_precise_exclusion_mask(landmarks, height, width)
        
        # Subtract exclusions from face mask
        foundation_mask = cv2.subtract(mask, exclusion_mask)
        
        # Use minimal morphological operations to stay within boundaries
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foundation_mask = cv2.morphologyEx(foundation_mask, cv2.MORPH_CLOSE, kernel_small)
        
        # Gentle smoothing to remove hard edges but stay precise
        foundation_mask = cv2.GaussianBlur(foundation_mask, (9, 9), 0)
        
        # Final boundary check to ensure we don't go outside face
        foundation_mask = apply_face_boundary_constraint(foundation_mask, landmarks, height, width)
        
        return foundation_mask
        
    except Exception as e:
        print(f"Error in precise face mask: {e}")
        return create_tight_fallback_mask(landmarks, height, width)

def create_precise_cheek_mask(landmarks, height, width):
    """Create precise cheek masks that stay within face boundaries"""
    cheek_mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Left cheek - precise points only
        left_cheek_points = normalize_landmarks(landmarks, height, width, LEFT_CHEEK)
        if left_cheek_points.size > 0 and len(left_cheek_points) > 2:
            left_cheek_hull = cv2.convexHull(left_cheek_points)
            cv2.fillPoly(cheek_mask, [left_cheek_hull], 255)
        
        # Right cheek - precise points only
        right_cheek_points = normalize_landmarks(landmarks, height, width, RIGHT_CHEEK)
        if right_cheek_points.size > 0 and len(right_cheek_points) > 2:
            right_cheek_hull = cv2.convexHull(right_cheek_points)
            cv2.fillPoly(cheek_mask, [right_cheek_hull], 255)
        
        # Nose sides
        nose_points = normalize_landmarks(landmarks, height, width, NOSE_SIDES)
        if nose_points.size > 0 and len(nose_points) > 2:
            nose_hull = cv2.convexHull(nose_points)
            cv2.fillPoly(cheek_mask, [nose_hull], 255)
        
        # Minimal expansion to connect areas without going outside
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cheek_mask = cv2.dilate(cheek_mask, kernel_tiny, iterations=1)
        
        return cheek_mask
        
    except Exception as e:
        print(f"Error creating precise cheek mask: {e}")
        return np.zeros((height, width), dtype=np.uint8)

def create_tight_face_boundary(landmarks, height, width):
    """Create a tight boundary to prevent foundation from going outside face"""
    boundary_mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Use the main face contour as the absolute boundary
        face_points = normalize_landmarks(landmarks, height, width, FACE_CONTOUR)
        if face_points.size > 0 and len(face_points) > 2:
            face_hull = cv2.convexHull(face_points)
            cv2.fillPoly(boundary_mask, [face_hull], 255)
        
        # Slightly contract the boundary to ensure we stay inside
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boundary_mask = cv2.erode(boundary_mask, kernel, iterations=1)
        
        return boundary_mask
        
    except Exception as e:
        print(f"Error creating tight boundary: {e}")
        # Fallback: create conservative oval
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        axes = (width // 3, height // 3)  # Conservative size
        cv2.ellipse(boundary_mask, center, axes, 0, 0, 360, 255, -1)
        return boundary_mask

def apply_face_boundary_constraint(mask, landmarks, height, width):
    """Ensure mask doesn't extend beyond face boundaries"""
    try:
        # Create tight face boundary
        face_boundary = create_tight_face_boundary(landmarks, height, width)
        
        # Apply boundary constraint
        constrained_mask = cv2.bitwise_and(mask, face_boundary)
        
        return constrained_mask
        
    except Exception as e:
        print(f"Error applying boundary constraint: {e}")
        return mask

def create_precise_exclusion_mask(landmarks, height, width):
    """Create precise exclusion masks"""
    exclusion_mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Eyes - precise exclusion
        left_eye_points = normalize_landmarks(landmarks, height, width, LEFT_EYE)
        right_eye_points = normalize_landmarks(landmarks, height, width, RIGHT_EYE)
        
        if left_eye_points.size > 0 and len(left_eye_points) > 2:
            left_eye_hull = cv2.convexHull(left_eye_points)
            cv2.fillPoly(exclusion_mask, [left_eye_hull], 255)
        
        if right_eye_points.size > 0 and len(right_eye_points) > 2:
            right_eye_hull = cv2.convexHull(right_eye_points)
            cv2.fillPoly(exclusion_mask, [right_eye_hull], 255)
        
        # Lips - precise exclusion
        lip_points = normalize_landmarks(landmarks, height, width, LIPS)
        if lip_points.size > 0 and len(lip_points) > 2:
            lip_hull = cv2.convexHull(lip_points)
            cv2.fillPoly(exclusion_mask, [lip_hull], 255)
        
        # No dilation for exclusions - keep them precise
        return exclusion_mask
        
    except Exception as e:
        print(f"Error creating exclusion mask: {e}")
        return np.zeros((height, width), dtype=np.uint8)

def create_tight_fallback_mask(landmarks, height, width):
    """Tight fallback mask that doesn't extend beyond reasonable face area"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Conservative oval in center
    center = (width // 2, height // 2)
    axes = (width // 4, height // 3)  # Much tighter than before
    
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # Gentle smoothing
    return cv2.GaussianBlur(mask, (15, 15), 0)

def apply_foundation_to_uploaded_image(src, shade='03', intensity=0.7):
    """
    Apply foundation to uploaded images with better face detection
    """
    print(f"🎨 Starting foundation application for uploaded image...")
    print(f"   Shade: {shade}, Intensity: {intensity}")
    
    if src is None or src.size == 0:
        print("❌ Invalid source image")
        return src
    
    # Check if foundation is enabled
    if not app_state.foundation_enabled:
        print("ℹ️ Foundation is disabled")
        return src
    
    height, width = src.shape[:2]
    print(f"📸 Image dimensions: {width}x{height}")
    
    # Detect landmarks for uploaded image (static mode)
    landmarks = detect_landmarks_uploaded_image(src)
    
    if landmarks is None:
        print("❌ No face detected - returning original image")
        return src
    
    print(f"✅ Face detected with {len(landmarks)} landmarks")
    
    try:
        # Get foundation color
        foundation_bgr = get_shade_bgr(shade)
        foundation_color = np.array(foundation_bgr, dtype=np.uint8)
        print(f"🎨 Foundation color: {foundation_color}")
        
        # Create precise face mask with boundary constraints
        foundation_mask = create_precise_face_mask(landmarks, height, width)
        
        # Check if mask is valid
        if foundation_mask is None or np.sum(foundation_mask) == 0:
            print("❌ No valid face mask created")
            return src
        
        mask_pixels = np.sum(foundation_mask > 0)
        print(f"✅ Face mask created with {mask_pixels} pixels")
        
        # Apply foundation with advanced blending
        result = advanced_skin_blend(src, foundation_color, foundation_mask, intensity)
        
        # Add subtle texture to avoid flat look
        result = add_skin_texture(result, src, foundation_mask, intensity)
        
        # Add shade info
        shade_info = FOUNDATION_SHADES.get(shade, FOUNDATION_SHADES['03'])
        cv2.putText(result, f"{shade_info['name']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"{shade_info['name']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        print("🎨 Foundation applied successfully to uploaded image!")
        return result
        
    except Exception as e:
        print(f"❌ Error applying foundation to uploaded image: {e}")
        import traceback
        traceback.print_exc()
        return src

def apply_foundation(src, shade='03', intensity=0.7):
    """
    Apply foundation for live stream
    """
    if src is None or src.size == 0:
        return src
    
    # Check if foundation is enabled
    if not app_state.foundation_enabled:
        return src
    
    height, width = src.shape[:2]
    
    # Detect landmarks for live stream
    landmarks = detect_landmarks(src, True)
    
    if landmarks is None:
        return src  # Return original if no face detected
    
    try:
        # Get foundation color
        foundation_bgr = get_shade_bgr(shade)
        foundation_color = np.array(foundation_bgr, dtype=np.uint8)
        
        # Create precise face mask with boundary constraints
        foundation_mask = create_precise_face_mask(landmarks, height, width)
        
        # Apply foundation with advanced blending
        result = advanced_skin_blend(src, foundation_color, foundation_mask, intensity)
        
        # Add subtle texture to avoid flat look
        result = add_skin_texture(result, src, foundation_mask, intensity)
        
        # Add shade info
        shade_info = FOUNDATION_SHADES.get(shade, FOUNDATION_SHADES['03'])
        cv2.putText(result, f"{shade_info['name']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"{shade_info['name']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return result
        
    except Exception as e:
        print(f"Error applying foundation: {e}")
        return src

def advanced_skin_blend(base, foundation_color, mask, intensity):
    """Advanced blending with precise mask application"""
    height, width = base.shape[:2]
    
    # Convert to float for high precision blending
    base_float = base.astype(np.float32) / 255.0
    foundation_float = np.full_like(base_float, foundation_color / 255.0)
    
    # Get mask as float with intensity - ensure it's clean
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = mask_float * intensity
    
    # Ensure mask doesn't have stray pixels
    mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0)
    
    # LAB Color Space Blending
    blended = lab_color_blend(base_float, foundation_float, mask_float)
    
    # Add subtle color variation for realism
    blended = add_color_variation(blended, base_float, mask_float)
    
    # Convert back to uint8
    result = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    
    return result

def lab_color_blend(base, foundation, mask):
    """Blend using LAB color space to preserve skin texture"""
    # Convert to LAB color space
    base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    foundation_lab = cv2.cvtColor(foundation, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    base_L, base_A, base_B = cv2.split(base_lab)
    found_L, found_A, found_B = cv2.split(foundation_lab)
    
    # Preserve original luminosity (skin texture)
    blended_A = base_A * (1 - mask) + found_A * mask
    blended_B = base_B * (1 - mask) + found_B * mask
    
    # Very subtle luminosity adjustment
    luminosity_blend_strength = 0.05
    blended_L = base_L * (1 - mask * luminosity_blend_strength) + found_L * mask * luminosity_blend_strength
    
    # Merge back to LAB
    blended_lab = cv2.merge([blended_L, blended_A, blended_B])
    
    # Convert back to BGR
    blended_bgr = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
    
    return blended_bgr

def add_color_variation(result_float, original_float, mask):
    """Add subtle color variation to avoid uniform flat color"""
    # Extract the original image's color variation
    original_hsv = cv2.cvtColor(original_float, cv2.COLOR_BGR2HSV)
    result_hsv = cv2.cvtColor(result_float, cv2.COLOR_BGR2HSV)
    
    # Keep original saturation and value variations
    original_H, original_S, original_V = cv2.split(original_hsv)
    result_H, result_S, result_V = cv2.split(result_hsv)
    
    # Blend saturation: keep some of original saturation variation
    saturation_blend = 0.3
    blended_S = result_S * (1 - saturation_blend) + original_S * saturation_blend
    
    # Blend value: keep some of original brightness variation  
    value_blend = 0.2
    blended_V = result_V * (1 - value_blend) + original_V * value_blend
    
    # Merge back
    blended_hsv = cv2.merge([result_H, blended_S, blended_V])
    blended_bgr = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2BGR)
    
    return blended_bgr

def add_skin_texture(blended, original, mask, intensity):
    """Add back subtle skin texture"""
    # Convert to grayscale for texture analysis
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blended_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Extract high-frequency details (skin texture)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    original_blur = cv2.filter2D(original_gray, -1, kernel)
    texture = original_gray - original_blur
    
    # Normalize texture
    texture = texture - np.mean(texture)
    texture = texture / (np.std(texture) + 1e-8)
    
    # Add subtle texture back to blended image
    texture_strength = 0.01 * intensity
    mask_float = mask.astype(np.float32) / 255.0
    
    result_with_texture = blended.astype(np.float32)
    
    for channel in range(3):
        result_with_texture[:, :, channel] += texture * texture_strength * mask_float * 255.0
    
    # Clip to valid range
    result_with_texture = np.clip(result_with_texture, 0, 255)
    
    return result_with_texture.astype(np.uint8)

# Makeup application function for different features
def apply_makeup(src, foundation_enabled=True, feature='foundation', lips_enabled=False, shade='03'):
    """
    Apply makeup based on the selected feature
    """
    if src is None or src.size == 0:
        return src
    
    result = src.copy()
    
    # Apply foundation if enabled and feature is foundation
    if foundation_enabled and feature == 'foundation':
        result = apply_foundation(result, shade=shade)
    
    return result