# Foundation shades database
FOUNDATION_SHADES = {
    '01': {'name': 'PORCELAIN', 'undertone': 'Cool', 'hex': '#F5E9DE', 'bgr': [222, 233, 245]},
    '02': {'name': 'IVORY', 'undertone': 'Neutral', 'hex': '#F0E2D1', 'bgr': [209, 226, 240]},
    '03': {'name': 'WARM IVORY', 'undertone': 'Warm', 'hex': '#E0C7B0', 'bgr': [176, 199, 224]},
    '04': {'name': 'SAND', 'undertone': 'Warm', 'hex': '#D5B390', 'bgr': [144, 179, 213]},
    '05': {'name': 'BEIGE', 'undertone': 'Warm', 'hex': '#C8A181', 'bgr': [129, 161, 200]},
    '06': {'name': 'CARAMEL', 'undertone': 'Warm', 'hex': '#AF8A65', 'bgr': [101, 138, 175]},
    '07': {'name': 'WALNUT', 'undertone': 'Warm', 'hex': '#976E4B', 'bgr': [75, 110, 151]},
    '08': {'name': 'COFFEE', 'undertone': 'Warm', 'hex': '#7A573F', 'bgr': [63, 87, 122]},
    '09': {'name': 'DEEP COCOA', 'undertone': 'Warm', 'hex': '#5D4037', 'bgr': [55, 64, 93]}
}

# Application state
class AppState:
    def __init__(self):
        self.current_shade = '03'
        self.current_intensity = 0.7  # Foundation intensity
        self.foundation_enabled = True  # Added this missing property
    
    def set_shade(self, shade_key):
        if shade_key in FOUNDATION_SHADES:
            self.current_shade = shade_key
            print(f"Foundation shade set to: {FOUNDATION_SHADES[shade_key]['name']}")
            return True
        return False
    
    def set_intensity(self, intensity):
        if 0.0 <= intensity <= 1.0:
            self.current_intensity = intensity
            print(f"Foundation intensity set to: {intensity}")
            return True
        return False
    
    def get_current_shade_info(self):
        return FOUNDATION_SHADES.get(self.current_shade, FOUNDATION_SHADES['03'])
    
    def get_available_shades(self):
        return FOUNDATION_SHADES
    
    def get_state(self):
        return {
            'current_shade': self.current_shade,
            'current_intensity': self.current_intensity,
            'foundation_enabled': self.foundation_enabled,  # Added this
            'shade_info': self.get_current_shade_info()
        }
    
    def reset(self):
        self.current_shade = '03'
        self.current_intensity = 0.7
        self.foundation_enabled = True  # Added this

# Global application state
app_state = AppState()