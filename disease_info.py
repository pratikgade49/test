def get_disease_info(disease_name):
    """Return detailed information about the plant disease"""
    
    # Disease information database
    disease_db = {
        'Apple___Apple_scab': {
            'description': 'Apple scab is a fungal disease that affects apple trees, causing dark, scaly lesions on leaves and fruit.',
            'symptoms': [
                'Dark, olive-green to black spots on leaves',
                'Scaly lesions on fruit surface',
                'Premature leaf drop',
                'Reduced fruit quality'
            ],
            'treatment': [
                'Apply fungicide sprays during growing season',
                'Remove fallen leaves and debris',
                'Prune for better air circulation',
                'Use disease-resistant apple varieties'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Ensure good air circulation',
                'Avoid overhead watering',
                'Regular sanitation practices'
            ],
            'severity': 'medium'
        },
        
        'Apple___Black_rot': {
            'description': 'Black rot is a serious fungal disease that affects apple trees, causing fruit rot and leaf spots.',
            'symptoms': [
                'Brown leaf spots with purple margins',
                'Black, mummified fruit',
                'Cankers on branches',
                'Premature fruit drop'
            ],
            'treatment': [
                'Remove infected fruit and branches immediately',
                'Apply copper-based fungicides',
                'Improve orchard sanitation',
                'Prune infected wood during dormant season'
            ],
            'prevention': [
                'Plant disease-resistant varieties',
                'Maintain proper tree spacing',
                'Regular pruning for air circulation',
                'Remove mummified fruit from trees'
            ],
            'severity': 'high'
        },
        
        'Tomato___Early_blight': {
            'description': 'Early blight is a common fungal disease affecting tomatoes, characterized by dark spots with concentric rings.',
            'symptoms': [
                'Dark spots with concentric rings on leaves',
                'Yellowing and browning of lower leaves',
                'Stem lesions near soil line',
                'Fruit spots with dark, sunken areas'
            ],
            'treatment': [
                'Apply fungicide containing chlorothalonil or copper',
                'Remove affected plant debris',
                'Improve air circulation around plants',
                'Water at soil level, not on foliage'
            ],
            'prevention': [
                'Rotate crops yearly',
                'Use drip irrigation instead of overhead watering',
                'Mulch around plants to prevent soil splash',
                'Space plants for good air circulation'
            ],
            'severity': 'medium'
        },
        
        'Tomato___Late_blight': {
            'description': 'Late blight is a devastating disease that can quickly destroy tomato crops, especially in cool, wet conditions.',
            'symptoms': [
                'Water-soaked spots on leaves that turn brown',
                'White fuzzy growth on leaf undersides',
                'Brown, firm lesions on fruit',
                'Rapid plant death in severe cases'
            ],
            'treatment': [
                'Apply preventive fungicides immediately',
                'Remove and destroy infected plants',
                'Avoid overhead watering',
                'Improve drainage and air circulation'
            ],
            'prevention': [
                'Use certified disease-free seeds and plants',
                'Avoid working with wet plants',
                'Provide adequate spacing between plants',
                'Apply preventive fungicide treatments'
            ],
            'severity': 'high'
        },
        
        'Potato___Late_blight': {
            'description': 'Late blight is a serious disease of potatoes that can cause complete crop loss if not managed properly.',
            'symptoms': [
                'Dark, water-soaked lesions on leaves',
                'White mold growth on leaf undersides',
                'Brown rot in potato tubers',
                'Foul odor from infected tubers'
            ],
            'treatment': [
                'Apply fungicides containing metalaxyl or chlorothalonil',
                'Remove infected plant material immediately',
                'Avoid harvesting wet tubers',
                'Cure harvested potatoes properly'
            ],
            'prevention': [
                'Plant certified seed potatoes',
                'Avoid overhead irrigation',
                'Hill potatoes properly to prevent exposure',
                'Monitor weather conditions for disease-favorable periods'
            ],
            'severity': 'high'
        },
        
        'Corn_(maize)___Northern_Leaf_Blight': {
            'description': 'Northern leaf blight is a fungal disease that causes long, elliptical lesions on corn leaves.',
            'symptoms': [
                'Long, grayish-green to tan lesions on leaves',
                'Lesions may have dark borders',
                'Premature leaf death',
                'Reduced grain yield'
            ],
            'treatment': [
                'Apply foliar fungicides if economically justified',
                'Use resistant corn hybrids',
                'Manage crop residue properly',
                'Rotate with non-host crops'
            ],
            'prevention': [
                'Plant resistant varieties',
                'Practice crop rotation',
                'Manage corn residue through tillage',
                'Monitor fields regularly during growing season'
            ],
            'severity': 'medium'
        }
    }
    
    # Handle variations in disease names
    clean_name = disease_name.strip()
    
    # Direct match
    if clean_name in disease_db:
        return disease_db[clean_name]
    
    # For diseases not in the database, provide generic information
    if 'healthy' in clean_name.lower():
        return {
            'description': 'The plant appears to be healthy with no visible signs of disease.',
            'symptoms': ['No disease symptoms detected'],
            'treatment': ['Continue regular care and monitoring', 'Maintain good growing conditions'],
            'prevention': ['Regular inspection', 'Proper nutrition', 'Adequate watering', 'Good air circulation'],
            'severity': 'none'
        }
    
    # Generic disease information
    return {
        'description': f'This appears to be {clean_name.replace("_", " ").replace("___", " - ")}. Please consult local agricultural experts for specific treatment advice.',
        'symptoms': ['Visible symptoms detected by AI analysis'],
        'treatment': [
            'Consult with local agricultural extension service',
            'Consider appropriate fungicide or treatment based on specific disease',
            'Remove affected plant material if severe',
            'Improve growing conditions'
        ],
        'prevention': [
            'Use disease-resistant varieties when available',
            'Maintain proper plant spacing for air circulation',
            'Practice crop rotation',
            'Keep plants healthy with proper nutrition and watering'
        ],
        'severity': 'medium'
    }
