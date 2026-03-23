from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os
import requests

app = Flask(__name__)

# Load crop recommendation model
crop_model = pickle.load(open('model.pkl', 'rb'))
crop_sc = pickle.load(open('standard_scaler.pkl', 'rb'))
crop_ms = pickle.load(open('minmax_scaler.pkl', 'rb'))

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

@app.route('/')
def home():
    return render_template("landing.html")

@app.route('/get-weather-by-location', methods=['POST'])
def get_weather_by_location():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        season = data.get('season')
        soil_type = data.get('soil_type')
        
        if not lat or not lon or not season or not soil_type:
            return {'error': 'Location, season, or soil type not provided'}, 400
        
        # Find nearest district based on coordinates
        nearest_district = find_nearest_district(lat, lon)
        
        if not nearest_district:
            return {'error': 'Could not determine your district'}, 400
        
        # Get average weather data from our dataset
        weather_data = get_average_weather_data(nearest_district['state'], nearest_district['district'], season, soil_type)
        
        if weather_data:
            return {
                'success': True,
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'rainfall': weather_data['rainfall'],
                'location': f"{nearest_district['district']}, {nearest_district['state']}",
                'season': season,
                'soil_type': soil_type,
                'climate_zone': weather_data.get('climate_zone', 'Unknown')
            }
        else:
            return {'error': 'Weather data not available for your location'}, 400
            
    except Exception as e:
        print(f"Weather API error: {e}")
        return {'error': 'Failed to fetch weather data'}, 500

def find_nearest_district(lat, lon):
    """Find the nearest district based on coordinates"""
    # Comprehensive District coordinates mapping for all Indian states and districts
    district_coords = {
        # Andhra Pradesh - All 13 districts
        "Anantapur": {"lat": 14.6819, "lon": 77.6006, "state": "Andhra Pradesh"},
        "Chittoor": {"lat": 13.2172, "lon": 79.1003, "state": "Andhra Pradesh"},
        "East Godavari": {"lat": 17.2403, "lon": 81.8040, "state": "Andhra Pradesh"},
        "Guntur": {"lat": 16.3067, "lon": 80.4365, "state": "Andhra Pradesh"},
        "Krishna": {"lat": 16.1755, "lon": 81.3311, "state": "Andhra Pradesh"},
        "Kurnool": {"lat": 15.8281, "lon": 78.0373, "state": "Andhra Pradesh"},
        "Nellore": {"lat": 14.4426, "lon": 79.9865, "state": "Andhra Pradesh"},
        "Prakasam": {"lat": 15.8167, "lon": 79.9833, "state": "Andhra Pradesh"},  # Addanki coordinates
        "Srikakulam": {"lat": 18.2949, "lon": 83.8938, "state": "Andhra Pradesh"},
        "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
        "Vizianagaram": {"lat": 18.1167, "lon": 83.4000, "state": "Andhra Pradesh"},
        "West Godavari": {"lat": 16.7150, "lon": 81.1000, "state": "Andhra Pradesh"},
        "YSR Kadapa": {"lat": 14.4673, "lon": 78.8242, "state": "Andhra Pradesh"},
        
        # Arunachal Pradesh - All 25 districts
        "Tawang": {"lat": 27.5858, "lon": 91.8644, "state": "Arunachal Pradesh"},
        "West Kameng": {"lat": 27.3389, "lon": 92.4069, "state": "Arunachal Pradesh"},
        "East Kameng": {"lat": 27.0833, "lon": 93.0167, "state": "Arunachal Pradesh"},
        "Papum Pare": {"lat": 27.1000, "lon": 93.7167, "state": "Arunachal Pradesh"},
        "Kurung Kumey": {"lat": 27.8333, "lon": 93.4167, "state": "Arunachal Pradesh"},
        "Kra Daadi": {"lat": 27.5333, "lon": 93.6167, "state": "Arunachal Pradesh"},
        "Lower Subansiri": {"lat": 27.5833, "lon": 93.8333, "state": "Arunachal Pradesh"},
        "Upper Subansiri": {"lat": 28.1000, "lon": 94.0167, "state": "Arunachal Pradesh"},
        "West Siang": {"lat": 28.1667, "lon": 94.6167, "state": "Arunachal Pradesh"},
        "East Siang": {"lat": 28.0667, "lon": 95.3333, "state": "Arunachal Pradesh"},
        "Siang": {"lat": 28.4167, "lon": 94.9167, "state": "Arunachal Pradesh"},
        "Upper Siang": {"lat": 28.7167, "lon": 95.0167, "state": "Arunachal Pradesh"},
        "Lower Siang": {"lat": 27.8833, "lon": 94.8167, "state": "Arunachal Pradesh"},
        "Lower Dibang Valley": {"lat": 28.2167, "lon": 95.7833, "state": "Arunachal Pradesh"},
        "Dibang Valley": {"lat": 28.6167, "lon": 95.8333, "state": "Arunachal Pradesh"},
        "Anjaw": {"lat": 28.3333, "lon": 96.2167, "state": "Arunachal Pradesh"},
        "Lohit": {"lat": 27.9167, "lon": 96.2167, "state": "Arunachal Pradesh"},
        "Namsai": {"lat": 27.7833, "lon": 95.7833, "state": "Arunachal Pradesh"},
        "Changlang": {"lat": 27.1500, "lon": 95.7333, "state": "Arunachal Pradesh"},
        "Tirap": {"lat": 27.1000, "lon": 95.3333, "state": "Arunachal Pradesh"},
        "Longding": {"lat": 26.7833, "lon": 95.2667, "state": "Arunachal Pradesh"},
        "Pakke Kessang": {"lat": 26.9167, "lon": 92.8833, "state": "Arunachal Pradesh"},
        "Kamle": {"lat": 27.4167, "lon": 93.5833, "state": "Arunachal Pradesh"},
        "Keyi Panyor": {"lat": 27.6167, "lon": 94.2833, "state": "Arunachal Pradesh"},
        "Lepa Rada": {"lat": 27.7833, "lon": 93.6167, "state": "Arunachal Pradesh"},
        
        # Assam - All 34 districts
        "Baksa": {"lat": 26.6833, "lon": 91.1167, "state": "Assam"},
        "Barpeta": {"lat": 26.3167, "lon": 91.0000, "state": "Assam"},
        "Biswanath": {"lat": 26.7500, "lon": 93.1500, "state": "Assam"},
        "Bongaigaon": {"lat": 26.4833, "lon": 90.5500, "state": "Assam"},
        "Cachar": {"lat": 24.8333, "lon": 92.7789, "state": "Assam"},
        "Charaideo": {"lat": 27.0167, "lon": 94.9167, "state": "Assam"},
        "Chirang": {"lat": 26.6167, "lon": 90.2833, "state": "Assam"},
        "Darrang": {"lat": 26.4500, "lon": 92.0333, "state": "Assam"},
        "Dhemaji": {"lat": 27.4833, "lon": 94.5833, "state": "Assam"},
        "Dhubri": {"lat": 26.0167, "lon": 89.9833, "state": "Assam"},
        "Dibrugarh": {"lat": 27.4728, "lon": 94.9120, "state": "Assam"},
        "Dima Hasao": {"lat": 25.4667, "lon": 93.0167, "state": "Assam"},
        "Goalpara": {"lat": 26.1667, "lon": 90.6167, "state": "Assam"},
        "Golaghat": {"lat": 26.5167, "lon": 93.9667, "state": "Assam"},
        "Hailakandi": {"lat": 24.6833, "lon": 92.5667, "state": "Assam"},
        "Hojai": {"lat": 26.0000, "lon": 92.8667, "state": "Assam"},
        "Jorhat": {"lat": 26.7509, "lon": 94.2037, "state": "Assam"},
        "Kamrup": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
        "Kamrup Metropolitan": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
        "Karbi Anglong": {"lat": 26.0167, "lon": 93.4333, "state": "Assam"},
        "Karimganj": {"lat": 24.8667, "lon": 92.3500, "state": "Assam"},
        "Kokrajhar": {"lat": 26.4000, "lon": 90.2667, "state": "Assam"},
        "Lakhimpur": {"lat": 27.2333, "lon": 94.1000, "state": "Assam"},
        "Majuli": {"lat": 27.0000, "lon": 94.2167, "state": "Assam"},
        "Morigaon": {"lat": 26.2500, "lon": 92.3500, "state": "Assam"},
        "Nagaon": {"lat": 26.3467, "lon": 92.6833, "state": "Assam"},
        "Nalbari": {"lat": 26.4500, "lon": 91.4333, "state": "Assam"},
        "Sivasagar": {"lat": 26.9833, "lon": 94.6333, "state": "Assam"},
        "Sonitpur": {"lat": 26.6333, "lon": 92.8000, "state": "Assam"},
        "South Salmara-Mankachar": {"lat": 25.8667, "lon": 89.8667, "state": "Assam"},
        "Tinsukia": {"lat": 27.5000, "lon": 95.3667, "state": "Assam"},
        "Udalguri": {"lat": 26.7500, "lon": 92.1000, "state": "Assam"},
        "West Karbi Anglong": {"lat": 25.8833, "lon": 92.8667, "state": "Assam"},
        
        # Bihar - All 38 districts
        "Araria": {"lat": 26.1478, "lon": 87.5081, "state": "Bihar"},
        "Arwal": {"lat": 25.2500, "lon": 84.6833, "state": "Bihar"},
        "Aurangabad": {"lat": 24.7500, "lon": 84.3667, "state": "Bihar"},
        "Banka": {"lat": 24.8833, "lon": 86.9167, "state": "Bihar"},
        "Begusarai": {"lat": 25.4167, "lon": 86.1333, "state": "Bihar"},
        "Bhagalpur": {"lat": 25.2425, "lon": 86.9842, "state": "Bihar"},
        "Bhojpur": {"lat": 25.5000, "lon": 84.4667, "state": "Bihar"},
        "Buxar": {"lat": 25.5667, "lon": 83.9833, "state": "Bihar"},
        "Darbhanga": {"lat": 26.1542, "lon": 85.8918, "state": "Bihar"},
        "East Champaran": {"lat": 26.6500, "lon": 84.9167, "state": "Bihar"},
        "Gaya": {"lat": 24.7914, "lon": 85.0002, "state": "Bihar"},
        "Gopalganj": {"lat": 26.4667, "lon": 84.4333, "state": "Bihar"},
        "Jamui": {"lat": 24.9167, "lon": 86.2167, "state": "Bihar"},
        "Jehanabad": {"lat": 25.2167, "lon": 84.9833, "state": "Bihar"},
        "Kaimur": {"lat": 25.0500, "lon": 83.6167, "state": "Bihar"},
        "Katihar": {"lat": 25.5333, "lon": 87.5833, "state": "Bihar"},
        "Khagaria": {"lat": 25.5000, "lon": 86.4833, "state": "Bihar"},
        "Kishanganj": {"lat": 26.1000, "lon": 87.9500, "state": "Bihar"},
        "Lakhisarai": {"lat": 25.1833, "lon": 86.0833, "state": "Bihar"},
        "Madhepura": {"lat": 25.9167, "lon": 86.7833, "state": "Bihar"},
        "Madhubani": {"lat": 26.3500, "lon": 85.9167, "state": "Bihar"},
        "Munger": {"lat": 25.3833, "lon": 86.4667, "state": "Bihar"},
        "Muzaffarpur": {"lat": 26.1209, "lon": 85.3647, "state": "Bihar"},
        "Nalanda": {"lat": 25.1333, "lon": 85.4500, "state": "Bihar"},
        "Nawada": {"lat": 24.8833, "lon": 85.5333, "state": "Bihar"},
        "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
        "Purnia": {"lat": 25.7771, "lon": 87.4753, "state": "Bihar"},
        "Rohtas": {"lat": 24.9500, "lon": 84.0167, "state": "Bihar"},
        "Saharsa": {"lat": 25.8833, "lon": 86.6000, "state": "Bihar"},
        "Samastipur": {"lat": 25.8667, "lon": 85.7833, "state": "Bihar"},
        "Saran": {"lat": 25.7500, "lon": 84.7500, "state": "Bihar"},
        "Sheikhpura": {"lat": 25.1333, "lon": 85.8500, "state": "Bihar"},
        "Sheohar": {"lat": 26.5167, "lon": 85.2833, "state": "Bihar"},
        "Sitamarhi": {"lat": 26.5833, "lon": 85.4833, "state": "Bihar"},
        "Siwan": {"lat": 26.2167, "lon": 84.3667, "state": "Bihar"},
        "Supaul": {"lat": 26.1167, "lon": 86.6000, "state": "Bihar"},
        "Vaishali": {"lat": 25.7333, "lon": 85.1333, "state": "Bihar"},
        "West Champaran": {"lat": 27.0333, "lon": 84.5000, "state": "Bihar"},
        
        # Chhattisgarh - All 28 districts
        "Balod": {"lat": 20.7333, "lon": 81.2000, "state": "Chhattisgarh"},
        "Baloda Bazar": {"lat": 21.6667, "lon": 82.1667, "state": "Chhattisgarh"},
        "Balrampur": {"lat": 23.1167, "lon": 83.2833, "state": "Chhattisgarh"},
        "Bastar": {"lat": 19.0667, "lon": 81.9500, "state": "Chhattisgarh"},
        "Bemetara": {"lat": 21.7167, "lon": 81.5333, "state": "Chhattisgarh"},
        "Bijapur": {"lat": 19.0833, "lon": 80.7167, "state": "Chhattisgarh"},
        "Bilaspur": {"lat": 22.0797, "lon": 82.1409, "state": "Chhattisgarh"},
        "Dantewada": {"lat": 18.9000, "lon": 81.3500, "state": "Chhattisgarh"},
        "Dhamtari": {"lat": 20.7000, "lon": 81.5500, "state": "Chhattisgarh"},
        "Durg": {"lat": 21.1900, "lon": 81.2849, "state": "Chhattisgarh"},
        "Gariaband": {"lat": 20.6167, "lon": 82.0667, "state": "Chhattisgarh"},
        "Gaurela Pendra Marwahi": {"lat": 22.7833, "lon": 81.6167, "state": "Chhattisgarh"},
        "Janjgir Champa": {"lat": 22.0167, "lon": 82.5833, "state": "Chhattisgarh"},
        "Jashpur": {"lat": 22.8833, "lon": 84.1333, "state": "Chhattisgarh"},
        "Kabirdham": {"lat": 22.1167, "lon": 81.2333, "state": "Chhattisgarh"},
        "Kanker": {"lat": 20.2667, "lon": 81.4833, "state": "Chhattisgarh"},
        "Kondagaon": {"lat": 19.5833, "lon": 81.6667, "state": "Chhattisgarh"},
        "Korba": {"lat": 22.3595, "lon": 82.7501, "state": "Chhattisgarh"},
        "Koriya": {"lat": 23.2833, "lon": 82.6833, "state": "Chhattisgarh"},
        "Mahasamund": {"lat": 21.1000, "lon": 82.1000, "state": "Chhattisgarh"},
        "Mungeli": {"lat": 22.0667, "lon": 81.6833, "state": "Chhattisgarh"},
        "Narayanpur": {"lat": 19.5167, "lon": 81.2000, "state": "Chhattisgarh"},
        "Raigarh": {"lat": 21.8833, "lon": 83.4000, "state": "Chhattisgarh"},
        "Raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
        "Rajnandgaon": {"lat": 21.0833, "lon": 81.0333, "state": "Chhattisgarh"},
        "Sukma": {"lat": 18.3833, "lon": 81.6667, "state": "Chhattisgarh"},
        "Surajpur": {"lat": 23.2167, "lon": 82.8667, "state": "Chhattisgarh"},
        "Surguja": {"lat": 23.1167, "lon": 83.2000, "state": "Chhattisgarh"},
        
        # Goa - All 2 districts
        "North Goa": {"lat": 15.5500, "lon": 73.9167, "state": "Goa"},
        "South Goa": {"lat": 15.2500, "lon": 74.0000, "state": "Goa"},
        
        # Gujarat - All 33 districts
        "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
        "Amreli": {"lat": 21.6000, "lon": 71.2167, "state": "Gujarat"},
        "Anand": {"lat": 22.5500, "lon": 72.9500, "state": "Gujarat"},
        "Aravalli": {"lat": 23.2500, "lon": 73.0333, "state": "Gujarat"},
        "Banaskantha": {"lat": 24.1667, "lon": 72.4333, "state": "Gujarat"},
        "Bharuch": {"lat": 21.7000, "lon": 72.9833, "state": "Gujarat"},
        "Bhavnagar": {"lat": 21.7645, "lon": 72.1519, "state": "Gujarat"},
        "Botad": {"lat": 22.1667, "lon": 71.6667, "state": "Gujarat"},
        "Chhota Udaipur": {"lat": 22.3000, "lon": 74.0167, "state": "Gujarat"},
        "Dahod": {"lat": 22.8333, "lon": 74.2500, "state": "Gujarat"},
        "Dang": {"lat": 20.7500, "lon": 73.7000, "state": "Gujarat"},
        "Devbhoomi Dwarka": {"lat": 22.2500, "lon": 69.0833, "state": "Gujarat"},
        "Gandhinagar": {"lat": 23.2167, "lon": 72.6833, "state": "Gujarat"},
        "Gir Somnath": {"lat": 20.9000, "lon": 70.4000, "state": "Gujarat"},
        "Jamnagar": {"lat": 22.4707, "lon": 70.0577, "state": "Gujarat"},
        "Junagadh": {"lat": 21.5222, "lon": 70.4579, "state": "Gujarat"},
        "Kutch": {"lat": 23.7337, "lon": 69.8597, "state": "Gujarat"},
        "Kheda": {"lat": 22.7500, "lon": 72.6833, "state": "Gujarat"},
        "Mahisagar": {"lat": 23.2167, "lon": 73.6000, "state": "Gujarat"},
        "Mehsana": {"lat": 23.5833, "lon": 72.3833, "state": "Gujarat"},
        "Morbi": {"lat": 22.8167, "lon": 70.8333, "state": "Gujarat"},
        "Narmada": {"lat": 21.8833, "lon": 73.4167, "state": "Gujarat"},
        "Navsari": {"lat": 20.9500, "lon": 72.9333, "state": "Gujarat"},
        "Panchmahal": {"lat": 22.8333, "lon": 73.6167, "state": "Gujarat"},
        "Patan": {"lat": 23.8500, "lon": 72.1167, "state": "Gujarat"},
        "Porbandar": {"lat": 21.6417, "lon": 69.6293, "state": "Gujarat"},
        "Rajkot": {"lat": 22.3039, "lon": 70.8022, "state": "Gujarat"},
        "Sabarkantha": {"lat": 23.5500, "lon": 73.0500, "state": "Gujarat"},
        "Surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
        "Surendranagar": {"lat": 22.7167, "lon": 71.6333, "state": "Gujarat"},
        "Tapi": {"lat": 21.1167, "lon": 73.4167, "state": "Gujarat"},
        "Vadodara": {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat"},
        "Valsad": {"lat": 20.6167, "lon": 72.9333, "state": "Gujarat"},
        
        # Haryana - All 22 districts
        "Ambala": {"lat": 30.3782, "lon": 76.7767, "state": "Haryana"},
        "Bhiwani": {"lat": 28.7833, "lon": 76.1333, "state": "Haryana"},
        "Charkhi Dadri": {"lat": 28.5833, "lon": 76.2667, "state": "Haryana"},
        "Faridabad": {"lat": 28.4089, "lon": 77.3178, "state": "Haryana"},
        "Fatehabad": {"lat": 29.5167, "lon": 75.4500, "state": "Haryana"},
        "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
        "Hisar": {"lat": 29.1492, "lon": 75.7217, "state": "Haryana"},
        "Jhajjar": {"lat": 28.6000, "lon": 76.6500, "state": "Haryana"},
        "Jind": {"lat": 29.3167, "lon": 76.3167, "state": "Haryana"},
        "Kaithal": {"lat": 29.8000, "lon": 76.4000, "state": "Haryana"},
        "Karnal": {"lat": 29.6857, "lon": 76.9905, "state": "Haryana"},
        "Kurukshetra": {"lat": 29.9667, "lon": 76.8333, "state": "Haryana"},
        "Mahendragarh": {"lat": 28.2833, "lon": 76.1500, "state": "Haryana"},
        "Nuh": {"lat": 28.1000, "lon": 77.0167, "state": "Haryana"},
        "Palwal": {"lat": 28.1500, "lon": 77.3333, "state": "Haryana"},
        "Panchkula": {"lat": 30.6833, "lon": 76.8500, "state": "Haryana"},
        "Panipat": {"lat": 29.3909, "lon": 76.9635, "state": "Haryana"},
        "Rewari": {"lat": 28.1833, "lon": 76.6167, "state": "Haryana"},
        "Rohtak": {"lat": 28.8833, "lon": 76.6000, "state": "Haryana"},
        "Sirsa": {"lat": 29.5333, "lon": 75.0167, "state": "Haryana"},
        "Sonipat": {"lat": 28.9833, "lon": 77.0167, "state": "Haryana"},
        "Yamunanagar": {"lat": 30.1333, "lon": 77.2833, "state": "Haryana"},
        
        # Himachal Pradesh - All 12 districts
        "Bilaspur": {"lat": 31.3333, "lon": 76.7500, "state": "Himachal Pradesh"},
        "Chamba": {"lat": 32.5500, "lon": 76.1167, "state": "Himachal Pradesh"},
        "Hamirpur": {"lat": 31.6833, "lon": 76.5167, "state": "Himachal Pradesh"},
        "Kangra": {"lat": 32.0998, "lon": 76.2695, "state": "Himachal Pradesh"},
        "Kinnaur": {"lat": 31.6000, "lon": 78.4500, "state": "Himachal Pradesh"},
        "Kullu": {"lat": 31.9578, "lon": 77.1101, "state": "Himachal Pradesh"},
        "Lahaul and Spiti": {"lat": 32.5667, "lon": 77.1500, "state": "Himachal Pradesh"},
        "Mandi": {"lat": 31.7084, "lon": 76.9319, "state": "Himachal Pradesh"},
        "Shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
        "Sirmaur": {"lat": 30.5500, "lon": 77.3000, "state": "Himachal Pradesh"},
        "Solan": {"lat": 30.9000, "lon": 77.1000, "state": "Himachal Pradesh"},
        "Una": {"lat": 31.4667, "lon": 76.2667, "state": "Himachal Pradesh"},
        
        # Jharkhand - All 24 districts
        "Bokaro": {"lat": 23.6693, "lon": 86.1511, "state": "Jharkhand"},
        "Chatra": {"lat": 24.2000, "lon": 84.8667, "state": "Jharkhand"},
        "Deoghar": {"lat": 24.4833, "lon": 86.7000, "state": "Jharkhand"},
        "Dhanbad": {"lat": 23.7957, "lon": 86.4304, "state": "Jharkhand"},
        "Dumka": {"lat": 24.2667, "lon": 87.2500, "state": "Jharkhand"},
        "East Singhbhum": {"lat": 22.8046, "lon": 86.2029, "state": "Jharkhand"},
        "Garhwa": {"lat": 24.1667, "lon": 83.8000, "state": "Jharkhand"},
        "Giridih": {"lat": 24.1833, "lon": 86.3000, "state": "Jharkhand"},
        "Godda": {"lat": 24.8333, "lon": 87.2167, "state": "Jharkhand"},
        "Gumla": {"lat": 23.0500, "lon": 84.5333, "state": "Jharkhand"},
        "Hazaribagh": {"lat": 23.9833, "lon": 85.3667, "state": "Jharkhand"},
        "Jamtara": {"lat": 23.9667, "lon": 86.8000, "state": "Jharkhand"},
        "Khunti": {"lat": 23.0833, "lon": 85.2833, "state": "Jharkhand"},
        "Koderma": {"lat": 24.4667, "lon": 85.5833, "state": "Jharkhand"},
        "Latehar": {"lat": 23.7333, "lon": 84.5000, "state": "Jharkhand"},
        "Lohardaga": {"lat": 23.4333, "lon": 84.6833, "state": "Jharkhand"},
        "Pakur": {"lat": 24.6333, "lon": 87.8500, "state": "Jharkhand"},
        "Palamu": {"lat": 24.0333, "lon": 84.0667, "state": "Jharkhand"},
        "Ramgarh": {"lat": 23.6167, "lon": 85.5167, "state": "Jharkhand"},
        "Ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
        "Sahebganj": {"lat": 25.2500, "lon": 87.6333, "state": "Jharkhand"},
        "Seraikela Kharsawan": {"lat": 22.6833, "lon": 85.9167, "state": "Jharkhand"},
        "Simdega": {"lat": 22.6167, "lon": 84.5167, "state": "Jharkhand"},
        "West Singhbhum": {"lat": 22.5667, "lon": 85.8167, "state": "Jharkhand"},
        
        # Karnataka - All 30 districts
        "Bagalkot": {"lat": 16.1833, "lon": 75.7000, "state": "Karnataka"},
        "Ballari": {"lat": 15.1394, "lon": 76.9214, "state": "Karnataka"},
        "Belagavi": {"lat": 15.8497, "lon": 74.4977, "state": "Karnataka"},
        "Bengaluru Rural": {"lat": 13.2846, "lon": 77.6735, "state": "Karnataka"},
        "Bengaluru Urban": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
        "Bidar": {"lat": 17.9167, "lon": 77.5167, "state": "Karnataka"},
        "Chamarajanagar": {"lat": 11.9167, "lon": 76.9500, "state": "Karnataka"},
        "Chikballapur": {"lat": 13.4333, "lon": 77.7333, "state": "Karnataka"},
        "Chikkamagaluru": {"lat": 13.3167, "lon": 75.7667, "state": "Karnataka"},
        "Chitradurga": {"lat": 14.2167, "lon": 76.4000, "state": "Karnataka"},
        "Dakshina Kannada": {"lat": 12.9141, "lon": 74.8560, "state": "Karnataka"},
        "Davanagere": {"lat": 14.4667, "lon": 75.9167, "state": "Karnataka"},
        "Dharwad": {"lat": 15.4500, "lon": 75.0167, "state": "Karnataka"},
        "Gadag": {"lat": 15.4167, "lon": 75.6333, "state": "Karnataka"},
        "Hassan": {"lat": 13.0000, "lon": 76.1000, "state": "Karnataka"},
        "Haveri": {"lat": 14.7833, "lon": 75.4000, "state": "Karnataka"},
        "Kalaburagi": {"lat": 17.3297, "lon": 76.8343, "state": "Karnataka"},
        "Kodagu": {"lat": 12.4167, "lon": 75.7333, "state": "Karnataka"},
        "Kolar": {"lat": 13.1333, "lon": 78.1333, "state": "Karnataka"},
        "Koppal": {"lat": 15.3500, "lon": 76.1500, "state": "Karnataka"},
        "Mandya": {"lat": 12.5167, "lon": 76.9000, "state": "Karnataka"},
        "Mysuru": {"lat": 12.2958, "lon": 76.6394, "state": "Karnataka"},
        "Raichur": {"lat": 16.2000, "lon": 77.3500, "state": "Karnataka"},
        "Ramanagara": {"lat": 12.7167, "lon": 77.2833, "state": "Karnataka"},
        "Shivamogga": {"lat": 13.9167, "lon": 75.5667, "state": "Karnataka"},
        "Tumakuru": {"lat": 13.3379, "lon": 77.1022, "state": "Karnataka"},
        "Udupi": {"lat": 13.3333, "lon": 74.7500, "state": "Karnataka"},
        "Uttara Kannada": {"lat": 14.7833, "lon": 74.6833, "state": "Karnataka"},
        "Vijayapura": {"lat": 16.8167, "lon": 75.7167, "state": "Karnataka"},
        "Yadgir": {"lat": 16.7667, "lon": 77.1333, "state": "Karnataka"},
        
        # Kerala - All 14 districts
        "Alappuzha": {"lat": 9.4981, "lon": 76.3388, "state": "Kerala"},
        "Ernakulam": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
        "Idukki": {"lat": 9.8500, "lon": 76.9667, "state": "Kerala"},
        "Kannur": {"lat": 11.8745, "lon": 75.3704, "state": "Kerala"},
        "Kasaragod": {"lat": 12.5000, "lon": 75.0000, "state": "Kerala"},
        "Kollam": {"lat": 8.8932, "lon": 76.6141, "state": "Kerala"},
        "Kottayam": {"lat": 9.5916, "lon": 76.5222, "state": "Kerala"},
        "Kozhikode": {"lat": 11.2588, "lon": 75.7804, "state": "Kerala"},
        "Malappuram": {"lat": 11.0500, "lon": 76.0833, "state": "Kerala"},
        "Palakkad": {"lat": 10.7667, "lon": 76.6500, "state": "Kerala"},
        "Pathanamthitta": {"lat": 9.2667, "lon": 76.7833, "state": "Kerala"},
        "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
        "Thrissur": {"lat": 10.5276, "lon": 76.2144, "state": "Kerala"},
        "Wayanad": {"lat": 11.6000, "lon": 76.0833, "state": "Kerala"},
        
        # Madhya Pradesh - All 52 districts
        "Agar Malwa": {"lat": 23.7167, "lon": 76.0167, "state": "Madhya Pradesh"},
        "Alirajpur": {"lat": 22.3000, "lon": 74.3667, "state": "Madhya Pradesh"},
        "Anuppur": {"lat": 23.1000, "lon": 81.6833, "state": "Madhya Pradesh"},
        "Ashoknagar": {"lat": 24.5667, "lon": 77.7333, "state": "Madhya Pradesh"},
        "Balaghat": {"lat": 21.8000, "lon": 80.1833, "state": "Madhya Pradesh"},
        "Barwani": {"lat": 22.0333, "lon": 74.9000, "state": "Madhya Pradesh"},
        "Betul": {"lat": 21.9000, "lon": 77.9000, "state": "Madhya Pradesh"},
        "Bhind": {"lat": 26.5667, "lon": 78.7833, "state": "Madhya Pradesh"},
        "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
        "Burhanpur": {"lat": 21.3000, "lon": 76.2333, "state": "Madhya Pradesh"},
        "Chachaura": {"lat": 24.2167, "lon": 78.9500, "state": "Madhya Pradesh"},
        "Chhatarpur": {"lat": 24.9167, "lon": 79.5833, "state": "Madhya Pradesh"},
        "Chhindwara": {"lat": 22.0667, "lon": 78.9333, "state": "Madhya Pradesh"},
        "Damoh": {"lat": 23.8333, "lon": 79.4333, "state": "Madhya Pradesh"},
        "Datia": {"lat": 25.6667, "lon": 78.4667, "state": "Madhya Pradesh"},
        "Dewas": {"lat": 22.9667, "lon": 76.0500, "state": "Madhya Pradesh"},
        "Dhar": {"lat": 22.6000, "lon": 75.3000, "state": "Madhya Pradesh"},
        "Dindori": {"lat": 22.9500, "lon": 81.0833, "state": "Madhya Pradesh"},
        "Guna": {"lat": 24.6500, "lon": 77.3167, "state": "Madhya Pradesh"},
        "Gwalior": {"lat": 26.2183, "lon": 78.1828, "state": "Madhya Pradesh"},
        "Harda": {"lat": 22.3333, "lon": 77.0833, "state": "Madhya Pradesh"},
        "Hoshangabad": {"lat": 22.7500, "lon": 77.7167, "state": "Madhya Pradesh"},
        "Indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
        "Jabalpur": {"lat": 23.1815, "lon": 79.9864, "state": "Madhya Pradesh"},
        "Jhabua": {"lat": 22.7667, "lon": 74.6000, "state": "Madhya Pradesh"},
        "Katni": {"lat": 23.8333, "lon": 80.4000, "state": "Madhya Pradesh"},
        "Khandwa": {"lat": 21.8333, "lon": 76.3500, "state": "Madhya Pradesh"},
        "Khargone": {"lat": 21.8167, "lon": 75.6167, "state": "Madhya Pradesh"},
        "Maihar": {"lat": 24.2667, "lon": 80.7500, "state": "Madhya Pradesh"},
        "Mandla": {"lat": 22.6000, "lon": 80.3667, "state": "Madhya Pradesh"},
        "Mandsaur": {"lat": 24.0667, "lon": 75.0667, "state": "Madhya Pradesh"},
        "Morena": {"lat": 26.5000, "lon": 78.0000, "state": "Madhya Pradesh"},
        "Narsinghpur": {"lat": 22.9500, "lon": 79.2000, "state": "Madhya Pradesh"},
        "Neemuch": {"lat": 24.4667, "lon": 74.8667, "state": "Madhya Pradesh"},
        "Niwari": {"lat": 25.6167, "lon": 79.0000, "state": "Madhya Pradesh"},
        "Panna": {"lat": 24.7167, "lon": 80.1833, "state": "Madhya Pradesh"},
        "Raisen": {"lat": 23.3167, "lon": 77.7833, "state": "Madhya Pradesh"},
        "Rajgarh": {"lat": 24.0000, "lon": 76.7333, "state": "Madhya Pradesh"},
        "Ratlam": {"lat": 23.3167, "lon": 75.0333, "state": "Madhya Pradesh"},
        "Rewa": {"lat": 24.5333, "lon": 81.3000, "state": "Madhya Pradesh"},
        "Sagar": {"lat": 23.8333, "lon": 78.7333, "state": "Madhya Pradesh"},
        "Satna": {"lat": 24.5833, "lon": 80.8333, "state": "Madhya Pradesh"},
        "Sehore": {"lat": 23.2000, "lon": 77.0833, "state": "Madhya Pradesh"},
        "Seoni": {"lat": 22.0833, "lon": 79.5333, "state": "Madhya Pradesh"},
        "Shahdol": {"lat": 23.3000, "lon": 81.3667, "state": "Madhya Pradesh"},
        "Shajapur": {"lat": 23.4167, "lon": 76.2667, "state": "Madhya Pradesh"},
        "Sheopur": {"lat": 25.6667, "lon": 76.7000, "state": "Madhya Pradesh"},
        "Shivpuri": {"lat": 25.4333, "lon": 77.6500, "state": "Madhya Pradesh"},
        "Sidhi": {"lat": 24.4167, "lon": 81.8833, "state": "Madhya Pradesh"},
        "Singrauli": {"lat": 24.2000, "lon": 82.6667, "state": "Madhya Pradesh"},
        "Tikamgarh": {"lat": 24.7333, "lon": 78.8333, "state": "Madhya Pradesh"},
        "Ujjain": {"lat": 23.1765, "lon": 75.7885, "state": "Madhya Pradesh"},
        "Umaria": {"lat": 23.5167, "lon": 80.8333, "state": "Madhya Pradesh"},
        "Vidisha": {"lat": 23.5167, "lon": 77.8000, "state": "Madhya Pradesh"},
        
        # Maharashtra - All 36 districts
        "Ahmednagar": {"lat": 19.0833, "lon": 74.7333, "state": "Maharashtra"},
        "Akola": {"lat": 20.7000, "lon": 77.0000, "state": "Maharashtra"},
        "Amravati": {"lat": 20.9333, "lon": 77.7500, "state": "Maharashtra"},
        "Aurangabad": {"lat": 19.8762, "lon": 75.3433, "state": "Maharashtra"},
        "Beed": {"lat": 18.9833, "lon": 75.7500, "state": "Maharashtra"},
        "Bhandara": {"lat": 21.1667, "lon": 79.6500, "state": "Maharashtra"},
        "Buldhana": {"lat": 20.5333, "lon": 76.1833, "state": "Maharashtra"},
        "Chandrapur": {"lat": 19.9500, "lon": 79.3000, "state": "Maharashtra"},
        "Dhule": {"lat": 20.9000, "lon": 74.7667, "state": "Maharashtra"},
        "Gadchiroli": {"lat": 20.1667, "lon": 80.0000, "state": "Maharashtra"},
        "Gondia": {"lat": 21.4500, "lon": 80.2000, "state": "Maharashtra"},
        "Hingoli": {"lat": 19.7167, "lon": 77.1500, "state": "Maharashtra"},
        "Jalgaon": {"lat": 21.0000, "lon": 75.5667, "state": "Maharashtra"},
        "Jalna": {"lat": 19.8333, "lon": 75.8833, "state": "Maharashtra"},
        "Kolhapur": {"lat": 16.7000, "lon": 74.2167, "state": "Maharashtra"},
        "Latur": {"lat": 18.4000, "lon": 76.5833, "state": "Maharashtra"},
        "Mumbai City": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
        "Mumbai Suburban": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
        "Nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
        "Nanded": {"lat": 19.1500, "lon": 77.3000, "state": "Maharashtra"},
        "Nandurbar": {"lat": 21.3667, "lon": 74.2333, "state": "Maharashtra"},
        "Nashik": {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra"},
        "Osmanabad": {"lat": 18.1833, "lon": 76.0333, "state": "Maharashtra"},
        "Palghar": {"lat": 19.6833, "lon": 72.7667, "state": "Maharashtra"},
        "Parbhani": {"lat": 19.2667, "lon": 76.7667, "state": "Maharashtra"},
        "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
        "Raigad": {"lat": 18.5167, "lon": 73.1833, "state": "Maharashtra"},
        "Ratnagiri": {"lat": 16.9833, "lon": 73.3000, "state": "Maharashtra"},
        "Sangli": {"lat": 16.8667, "lon": 74.5667, "state": "Maharashtra"},
        "Satara": {"lat": 17.6833, "lon": 74.0167, "state": "Maharashtra"},
        "Sindhudurg": {"lat": 16.0000, "lon": 73.5000, "state": "Maharashtra"},
        "Solapur": {"lat": 17.6599, "lon": 75.9064, "state": "Maharashtra"},
        "Thane": {"lat": 19.2183, "lon": 72.9781, "state": "Maharashtra"},
        "Wardha": {"lat": 20.7333, "lon": 78.6000, "state": "Maharashtra"},
        "Washim": {"lat": 20.1000, "lon": 77.1333, "state": "Maharashtra"},
        "Yavatmal": {"lat": 20.3833, "lon": 78.1333, "state": "Maharashtra"},
        
        # Manipur - All 16 districts
        "Bishnupur": {"lat": 24.6167, "lon": 93.7833, "state": "Manipur"},
        "Chandel": {"lat": 24.3167, "lon": 94.0167, "state": "Manipur"},
        "Churachandpur": {"lat": 24.3333, "lon": 93.6833, "state": "Manipur"},
        "Imphal East": {"lat": 24.8167, "lon": 93.9500, "state": "Manipur"},
        "Imphal West": {"lat": 24.8167, "lon": 93.9500, "state": "Manipur"},
        "Jiribam": {"lat": 24.8000, "lon": 93.1167, "state": "Manipur"},
        "Kakching": {"lat": 24.4833, "lon": 93.9833, "state": "Manipur"},
        "Kamjong": {"lat": 24.9167, "lon": 94.3167, "state": "Manipur"},
        "Kangpokpi": {"lat": 25.1833, "lon": 93.9833, "state": "Manipur"},
        "Noney": {"lat": 25.1167, "lon": 93.7833, "state": "Manipur"},
        "Pherzawl": {"lat": 24.1667, "lon": 93.2500, "state": "Manipur"},
        "Senapati": {"lat": 25.2667, "lon": 94.0167, "state": "Manipur"},
        "Tamenglong": {"lat": 24.9833, "lon": 93.5000, "state": "Manipur"},
        "Tengnoupal": {"lat": 24.3833, "lon": 94.2167, "state": "Manipur"},
        "Thoubal": {"lat": 24.6333, "lon": 94.0167, "state": "Manipur"},
        "Ukhrul": {"lat": 25.0833, "lon": 94.3667, "state": "Manipur"},
        
        # Meghalaya - All 11 districts
        "East Garo Hills": {"lat": 25.5167, "lon": 90.6333, "state": "Meghalaya"},
        "East Jaintia Hills": {"lat": 25.4833, "lon": 92.3333, "state": "Meghalaya"},
        "East Khasi Hills": {"lat": 25.5667, "lon": 91.8833, "state": "Meghalaya"},
        "North Garo Hills": {"lat": 25.9167, "lon": 90.6167, "state": "Meghalaya"},
        "Ri Bhoi": {"lat": 25.8333, "lon": 91.9167, "state": "Meghalaya"},
        "South Garo Hills": {"lat": 25.1833, "lon": 90.4833, "state": "Meghalaya"},
        "South West Garo Hills": {"lat": 25.3000, "lon": 90.0167, "state": "Meghalaya"},
        "South West Khasi Hills": {"lat": 25.3167, "lon": 91.2833, "state": "Meghalaya"},
        "West Garo Hills": {"lat": 25.5833, "lon": 90.2167, "state": "Meghalaya"},
        "West Jaintia Hills": {"lat": 25.4500, "lon": 92.1833, "state": "Meghalaya"},
        "West Khasi Hills": {"lat": 25.5500, "lon": 91.2833, "state": "Meghalaya"},
        
        # Mizoram - All 11 districts
        "Aizawl": {"lat": 23.7271, "lon": 92.7176, "state": "Mizoram"},
        "Champhai": {"lat": 23.4500, "lon": 93.3167, "state": "Mizoram"},
        "Hnahthial": {"lat": 22.9667, "lon": 92.9833, "state": "Mizoram"},
        "Khawzawl": {"lat": 23.3833, "lon": 93.1167, "state": "Mizoram"},
        "Kolasib": {"lat": 24.2167, "lon": 92.6833, "state": "Mizoram"},
        "Lawngtlai": {"lat": 22.5167, "lon": 92.8833, "state": "Mizoram"},
        "Lunglei": {"lat": 22.8833, "lon": 92.7333, "state": "Mizoram"},
        "Mamit": {"lat": 23.9167, "lon": 92.4833, "state": "Mizoram"},
        "Saiha": {"lat": 22.4833, "lon": 92.9833, "state": "Mizoram"},
        "Saitual": {"lat": 23.8333, "lon": 93.0000, "state": "Mizoram"},
        "Serchhip": {"lat": 23.3000, "lon": 92.8333, "state": "Mizoram"},
        
        # Nagaland - All 12 districts
        "Dimapur": {"lat": 25.9167, "lon": 93.7333, "state": "Nagaland"},
        "Kiphire": {"lat": 25.8833, "lon": 94.8333, "state": "Nagaland"},
        "Kohima": {"lat": 25.6667, "lon": 94.1167, "state": "Nagaland"},
        "Longleng": {"lat": 26.4167, "lon": 94.5333, "state": "Nagaland"},
        "Mokokchung": {"lat": 26.3167, "lon": 94.5167, "state": "Nagaland"},
        "Mon": {"lat": 26.7167, "lon": 95.0167, "state": "Nagaland"},
        "Noklak": {"lat": 26.6167, "lon": 95.1167, "state": "Nagaland"},
        "Peren": {"lat": 25.5000, "lon": 93.7333, "state": "Nagaland"},
        "Phek": {"lat": 25.6667, "lon": 94.4333, "state": "Nagaland"},
        "Tuensang": {"lat": 26.2667, "lon": 94.8167, "state": "Nagaland"},
        "Wokha": {"lat": 26.0833, "lon": 94.2667, "state": "Nagaland"},
        "Zunheboto": {"lat": 25.9667, "lon": 94.5167, "state": "Nagaland"},
        
        # Odisha - All 30 districts
        "Angul": {"lat": 20.8333, "lon": 85.1000, "state": "Odisha"},
        "Balangir": {"lat": 20.7000, "lon": 83.4833, "state": "Odisha"},
        "Balasore": {"lat": 21.4833, "lon": 86.9333, "state": "Odisha"},
        "Bargarh": {"lat": 21.3333, "lon": 83.6167, "state": "Odisha"},
        "Bhadrak": {"lat": 21.0500, "lon": 86.5167, "state": "Odisha"},
        "Boudh": {"lat": 20.4333, "lon": 84.3167, "state": "Odisha"},
        "Cuttack": {"lat": 20.4625, "lon": 85.8828, "state": "Odisha"},
        "Deogarh": {"lat": 21.5333, "lon": 84.7333, "state": "Odisha"},
        "Dhenkanal": {"lat": 20.6667, "lon": 85.6000, "state": "Odisha"},
        "Gajapati": {"lat": 18.8667, "lon": 84.1667, "state": "Odisha"},
        "Ganjam": {"lat": 19.3919, "lon": 84.8803, "state": "Odisha"},
        "Jagatsinghpur": {"lat": 20.2500, "lon": 86.1667, "state": "Odisha"},
        "Jajpur": {"lat": 20.8333, "lon": 86.3333, "state": "Odisha"},
        "Jharsuguda": {"lat": 21.8500, "lon": 84.0167, "state": "Odisha"},
        "Kalahandi": {"lat": 19.9167, "lon": 83.1667, "state": "Odisha"},
        "Kandhamal": {"lat": 20.1333, "lon": 84.1333, "state": "Odisha"},
        "Kendrapara": {"lat": 20.5000, "lon": 86.4167, "state": "Odisha"},
        "Kendujhar": {"lat": 21.6333, "lon": 85.5833, "state": "Odisha"},
        "Khordha": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
        "Koraput": {"lat": 18.8167, "lon": 82.7167, "state": "Odisha"},
        "Malkangiri": {"lat": 18.3500, "lon": 81.9000, "state": "Odisha"},
        "Mayurbhanj": {"lat": 21.9333, "lon": 86.7333, "state": "Odisha"},
        "Nabarangpur": {"lat": 19.2333, "lon": 82.5500, "state": "Odisha"},
        "Nayagarh": {"lat": 20.1333, "lon": 85.1000, "state": "Odisha"},
        "Nuapada": {"lat": 20.7667, "lon": 82.6167, "state": "Odisha"},
        "Puri": {"lat": 19.8135, "lon": 85.8312, "state": "Odisha"},
        "Rayagada": {"lat": 19.1667, "lon": 83.4167, "state": "Odisha"},
        "Sambalpur": {"lat": 21.4667, "lon": 83.9667, "state": "Odisha"},
        "Subarnapur": {"lat": 20.8333, "lon": 83.9000, "state": "Odisha"},
        "Sundargarh": {"lat": 22.1167, "lon": 84.0167, "state": "Odisha"},
        
        # Punjab - All 23 districts
        "Amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab"},
        "Barnala": {"lat": 30.3833, "lon": 75.5500, "state": "Punjab"},
        "Bathinda": {"lat": 30.2110, "lon": 74.9455, "state": "Punjab"},
        "Faridkot": {"lat": 30.6667, "lon": 74.7500, "state": "Punjab"},
        "Fatehgarh Sahib": {"lat": 30.6500, "lon": 76.4000, "state": "Punjab"},
        "Fazilka": {"lat": 30.4000, "lon": 74.0333, "state": "Punjab"},
        "Ferozepur": {"lat": 30.9293, "lon": 74.6132, "state": "Punjab"},
        "Gurdaspur": {"lat": 32.0333, "lon": 75.4000, "state": "Punjab"},
        "Hoshiarpur": {"lat": 31.5333, "lon": 75.9167, "state": "Punjab"},
        "Jalandhar": {"lat": 31.3260, "lon": 75.5762, "state": "Punjab"},
        "Kapurthala": {"lat": 31.3833, "lon": 75.3833, "state": "Punjab"},
        "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab"},
        "Malerkotla": {"lat": 30.5333, "lon": 75.8833, "state": "Punjab"},
        "Mansa": {"lat": 29.9833, "lon": 75.3833, "state": "Punjab"},
        "Moga": {"lat": 30.8167, "lon": 75.1667, "state": "Punjab"},
        "Muktsar": {"lat": 30.4667, "lon": 74.5167, "state": "Punjab"},
        "Pathankot": {"lat": 32.2667, "lon": 75.6500, "state": "Punjab"},
        "Patiala": {"lat": 30.3398, "lon": 76.3869, "state": "Punjab"},
        "Rupnagar": {"lat": 30.9667, "lon": 76.5333, "state": "Punjab"},
        "Sahibzada Ajit Singh Nagar": {"lat": 30.7046, "lon": 76.7179, "state": "Punjab"},
        "Sangrur": {"lat": 30.2500, "lon": 75.8333, "state": "Punjab"},
        "Shaheed Bhagat Singh Nagar": {"lat": 31.1000, "lon": 76.2167, "state": "Punjab"},
        "Tarn Taran": {"lat": 31.4500, "lon": 74.9333, "state": "Punjab"},
        
        # Rajasthan - All 33 districts
        "Ajmer": {"lat": 26.4499, "lon": 74.6399, "state": "Rajasthan"},
        "Alwar": {"lat": 27.5530, "lon": 76.6346, "state": "Rajasthan"},
        "Banswara": {"lat": 23.5500, "lon": 74.4333, "state": "Rajasthan"},
        "Baran": {"lat": 25.1000, "lon": 76.5167, "state": "Rajasthan"},
        "Barmer": {"lat": 25.7500, "lon": 71.4000, "state": "Rajasthan"},
        "Bharatpur": {"lat": 27.2167, "lon": 77.5000, "state": "Rajasthan"},
        "Bhilwara": {"lat": 25.3500, "lon": 74.6333, "state": "Rajasthan"},
        "Bikaner": {"lat": 28.0229, "lon": 73.3119, "state": "Rajasthan"},
        "Bundi": {"lat": 25.4333, "lon": 75.6333, "state": "Rajasthan"},
        "Chittorgarh": {"lat": 24.8833, "lon": 74.6167, "state": "Rajasthan"},
        "Churu": {"lat": 28.3000, "lon": 74.9667, "state": "Rajasthan"},
        "Dausa": {"lat": 26.8833, "lon": 76.3333, "state": "Rajasthan"},
        "Dholpur": {"lat": 26.7000, "lon": 77.9000, "state": "Rajasthan"},
        "Dungarpur": {"lat": 23.8333, "lon": 73.7167, "state": "Rajasthan"},
        "Hanumangarh": {"lat": 29.5833, "lon": 74.3167, "state": "Rajasthan"},
        "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
        "Jaisalmer": {"lat": 26.9167, "lon": 70.9167, "state": "Rajasthan"},
        "Jalore": {"lat": 25.3500, "lon": 72.6167, "state": "Rajasthan"},
        "Jhalawar": {"lat": 24.5833, "lon": 76.1667, "state": "Rajasthan"},
        "Jhunjhunu": {"lat": 28.1167, "lon": 75.4000, "state": "Rajasthan"},
        "Jodhpur": {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan"},
        "Karauli": {"lat": 26.5000, "lon": 77.0167, "state": "Rajasthan"},
        "Kota": {"lat": 25.2138, "lon": 75.8648, "state": "Rajasthan"},
        "Nagaur": {"lat": 27.2000, "lon": 73.7333, "state": "Rajasthan"},
        "Pali": {"lat": 25.7667, "lon": 73.3167, "state": "Rajasthan"},
        "Pratapgarh": {"lat": 24.0333, "lon": 74.7833, "state": "Rajasthan"},
        "Rajsamand": {"lat": 25.0667, "lon": 73.8833, "state": "Rajasthan"},
        "Sawai Madhopur": {"lat": 26.0167, "lon": 76.3500, "state": "Rajasthan"},
        "Sikar": {"lat": 27.6167, "lon": 75.1500, "state": "Rajasthan"},
        "Sirohi": {"lat": 24.8833, "lon": 72.8667, "state": "Rajasthan"},
        "Sri Ganganagar": {"lat": 29.9167, "lon": 73.8833, "state": "Rajasthan"},
        "Tonk": {"lat": 26.1667, "lon": 75.7833, "state": "Rajasthan"},
        "Udaipur": {"lat": 24.5854, "lon": 73.7125, "state": "Rajasthan"},
        
        # Sikkim - All 4 districts
        "East Sikkim": {"lat": 27.3333, "lon": 88.6167, "state": "Sikkim"},
        "North Sikkim": {"lat": 27.7333, "lon": 88.4667, "state": "Sikkim"},
        "South Sikkim": {"lat": 27.1667, "lon": 88.4167, "state": "Sikkim"},
        "West Sikkim": {"lat": 27.3167, "lon": 88.2167, "state": "Sikkim"},
        
        # Tamil Nadu - All 38 districts
        "Ariyalur": {"lat": 11.1333, "lon": 79.0833, "state": "Tamil Nadu"},
        "Chengalpattu": {"lat": 12.6833, "lon": 79.9833, "state": "Tamil Nadu"},
        "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
        "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
        "Cuddalore": {"lat": 11.7500, "lon": 79.7667, "state": "Tamil Nadu"},
        "Dharmapuri": {"lat": 12.1167, "lon": 78.1667, "state": "Tamil Nadu"},
        "Dindigul": {"lat": 10.3667, "lon": 77.9833, "state": "Tamil Nadu"},
        "Erode": {"lat": 11.3333, "lon": 77.7167, "state": "Tamil Nadu"},
        "Kallakurichi": {"lat": 11.7333, "lon": 78.9667, "state": "Tamil Nadu"},
        "Kanchipuram": {"lat": 12.8333, "lon": 79.7000, "state": "Tamil Nadu"},
        "Kanyakumari": {"lat": 8.0833, "lon": 77.5500, "state": "Tamil Nadu"},
        "Karur": {"lat": 10.9500, "lon": 78.0833, "state": "Tamil Nadu"},
        "Krishnagiri": {"lat": 12.5167, "lon": 78.2167, "state": "Tamil Nadu"},
        "Madurai": {"lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu"},
        "Mayiladuthurai": {"lat": 11.1000, "lon": 79.6500, "state": "Tamil Nadu"},
        "Nagapattinam": {"lat": 10.7667, "lon": 79.8333, "state": "Tamil Nadu"},
        "Namakkal": {"lat": 11.2167, "lon": 78.1667, "state": "Tamil Nadu"},
        "Nilgiris": {"lat": 11.4000, "lon": 76.7000, "state": "Tamil Nadu"},
        "Perambalur": {"lat": 11.2333, "lon": 78.8833, "state": "Tamil Nadu"},
        "Pudukkottai": {"lat": 10.3833, "lon": 78.8167, "state": "Tamil Nadu"},
        "Ramanathapuram": {"lat": 9.3667, "lon": 78.8333, "state": "Tamil Nadu"},
        "Ranipet": {"lat": 12.9333, "lon": 79.3333, "state": "Tamil Nadu"},
        "Salem": {"lat": 11.6643, "lon": 78.1460, "state": "Tamil Nadu"},
        "Sivaganga": {"lat": 9.8500, "lon": 78.4833, "state": "Tamil Nadu"},
        "Tenkasi": {"lat": 8.9500, "lon": 77.3167, "state": "Tamil Nadu"},
        "Thanjavur": {"lat": 10.7833, "lon": 79.1333, "state": "Tamil Nadu"},
        "Theni": {"lat": 10.0000, "lon": 77.4833, "state": "Tamil Nadu"},
        "Thoothukudi": {"lat": 8.8000, "lon": 78.1333, "state": "Tamil Nadu"},
        "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047, "state": "Tamil Nadu"},
        "Tirunelveli": {"lat": 8.7139, "lon": 77.7567, "state": "Tamil Nadu"},
        "Tirupattur": {"lat": 12.5000, "lon": 78.5667, "state": "Tamil Nadu"},
        "Tiruppur": {"lat": 11.1000, "lon": 77.3500, "state": "Tamil Nadu"},
        "Tiruvallur": {"lat": 13.1500, "lon": 79.9000, "state": "Tamil Nadu"},
        "Tiruvannamalai": {"lat": 12.2333, "lon": 79.0667, "state": "Tamil Nadu"},
        "Tiruvarur": {"lat": 10.7667, "lon": 79.6333, "state": "Tamil Nadu"},
        "Vellore": {"lat": 12.9165, "lon": 79.1325, "state": "Tamil Nadu"},
        "Viluppuram": {"lat": 11.9333, "lon": 79.5000, "state": "Tamil Nadu"},
        "Virudhunagar": {"lat": 9.5833, "lon": 77.9500, "state": "Tamil Nadu"},
        
        # Telangana - All 33 districts
        "Adilabad": {"lat": 19.6667, "lon": 78.5333, "state": "Telangana"},
        "Bhadradri Kothagudem": {"lat": 17.5500, "lon": 80.6167, "state": "Telangana"},
        "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
        "Jagtial": {"lat": 18.7833, "lon": 78.9167, "state": "Telangana"},
        "Jangaon": {"lat": 17.7167, "lon": 79.1667, "state": "Telangana"},
        "Jayashankar Bhupalapally": {"lat": 18.0000, "lon": 79.9000, "state": "Telangana"},
        "Jogulamba Gadwal": {"lat": 16.2167, "lon": 77.8000, "state": "Telangana"},
        "Kamareddy": {"lat": 18.3167, "lon": 78.3333, "state": "Telangana"},
        "Karimnagar": {"lat": 18.4333, "lon": 79.1333, "state": "Telangana"},
        "Khammam": {"lat": 17.2473, "lon": 80.1514, "state": "Telangana"},
        "Komaram Bheem Asifabad": {"lat": 19.3500, "lon": 79.3000, "state": "Telangana"},
        "Mahabubabad": {"lat": 17.6000, "lon": 80.0000, "state": "Telangana"},
        "Mahabubnagar": {"lat": 16.7333, "lon": 77.9833, "state": "Telangana"},
        "Mancherial": {"lat": 18.8667, "lon": 79.4500, "state": "Telangana"},
        "Medak": {"lat": 18.0500, "lon": 78.2667, "state": "Telangana"},
        "Medchal Malkajgiri": {"lat": 17.6167, "lon": 78.4833, "state": "Telangana"},
        "Mulugu": {"lat": 18.1833, "lon": 79.9333, "state": "Telangana"},
        "Nagarkurnool": {"lat": 16.4833, "lon": 78.3167, "state": "Telangana"},
        "Nalgonda": {"lat": 17.0500, "lon": 79.2667, "state": "Telangana"},
        "Narayanpet": {"lat": 16.7500, "lon": 77.5000, "state": "Telangana"},
        "Nirmal": {"lat": 19.0833, "lon": 78.3500, "state": "Telangana"},
        "Nizamabad": {"lat": 18.6725, "lon": 78.0941, "state": "Telangana"},
        "Peddapalli": {"lat": 18.6167, "lon": 79.3667, "state": "Telangana"},
        "Rajanna Sircilla": {"lat": 18.3833, "lon": 78.8167, "state": "Telangana"},
        "Rangareddy": {"lat": 17.3167, "lon": 78.2167, "state": "Telangana"},
        "Sangareddy": {"lat": 17.6167, "lon": 78.0833, "state": "Telangana"},
        "Siddipet": {"lat": 18.1000, "lon": 78.8500, "state": "Telangana"},
        "Suryapet": {"lat": 17.1333, "lon": 79.6167, "state": "Telangana"},
        "Vikarabad": {"lat": 17.3333, "lon": 77.9000, "state": "Telangana"},
        "Wanaparthy": {"lat": 16.3667, "lon": 78.0667, "state": "Telangana"},
        "Warangal Rural": {"lat": 17.9689, "lon": 79.5941, "state": "Telangana"},
        "Warangal Urban": {"lat": 17.9689, "lon": 79.5941, "state": "Telangana"},
        "Yadadri Bhuvanagiri": {"lat": 17.5500, "lon": 78.8833, "state": "Telangana"},
        
        # Tripura - All 8 districts
        "Dhalai": {"lat": 23.8333, "lon": 91.9167, "state": "Tripura"},
        "Gomati": {"lat": 23.5167, "lon": 91.4167, "state": "Tripura"},
        "Khowai": {"lat": 24.0667, "lon": 91.6000, "state": "Tripura"},
        "North Tripura": {"lat": 24.1833, "lon": 92.1667, "state": "Tripura"},
        "Sepahijala": {"lat": 23.7333, "lon": 91.3667, "state": "Tripura"},
        "South Tripura": {"lat": 23.1667, "lon": 91.4333, "state": "Tripura"},
        "Unakoti": {"lat": 24.3167, "lon": 92.0167, "state": "Tripura"},
        "West Tripura": {"lat": 23.8333, "lon": 91.2833, "state": "Tripura"},
        
        # Uttar Pradesh - All 75 districts
        "Agra": {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh"},
        "Aligarh": {"lat": 27.8833, "lon": 78.0833, "state": "Uttar Pradesh"},
        "Ambedkar Nagar": {"lat": 26.4000, "lon": 83.1833, "state": "Uttar Pradesh"},
        "Amethi": {"lat": 26.1500, "lon": 81.8167, "state": "Uttar Pradesh"},
        "Amroha": {"lat": 28.9000, "lon": 78.4667, "state": "Uttar Pradesh"},
        "Auraiya": {"lat": 26.4667, "lon": 79.5000, "state": "Uttar Pradesh"},
        "Ayodhya": {"lat": 26.8000, "lon": 82.2000, "state": "Uttar Pradesh"},
        "Azamgarh": {"lat": 26.0667, "lon": 83.1833, "state": "Uttar Pradesh"},
        "Baghpat": {"lat": 28.9500, "lon": 77.2167, "state": "Uttar Pradesh"},
        "Bahraich": {"lat": 27.5667, "lon": 81.6000, "state": "Uttar Pradesh"},
        "Ballia": {"lat": 25.7500, "lon": 84.1500, "state": "Uttar Pradesh"},
        "Balrampur": {"lat": 27.4333, "lon": 82.1833, "state": "Uttar Pradesh"},
        "Banda": {"lat": 25.4833, "lon": 80.3333, "state": "Uttar Pradesh"},
        "Barabanki": {"lat": 26.9167, "lon": 81.2000, "state": "Uttar Pradesh"},
        "Bareilly": {"lat": 28.3670, "lon": 79.4304, "state": "Uttar Pradesh"},
        "Basti": {"lat": 26.8000, "lon": 82.7333, "state": "Uttar Pradesh"},
        "Bhadohi": {"lat": 25.3833, "lon": 82.5667, "state": "Uttar Pradesh"},
        "Bijnor": {"lat": 29.3667, "lon": 78.1333, "state": "Uttar Pradesh"},
        "Budaun": {"lat": 28.0333, "lon": 79.1167, "state": "Uttar Pradesh"},
        "Bulandshahr": {"lat": 28.4000, "lon": 77.8500, "state": "Uttar Pradesh"},
        "Chandauli": {"lat": 25.2667, "lon": 83.2667, "state": "Uttar Pradesh"},
        "Chitrakoot": {"lat": 25.2000, "lon": 80.9000, "state": "Uttar Pradesh"},
        "Deoria": {"lat": 26.5000, "lon": 83.7833, "state": "Uttar Pradesh"},
        "Etah": {"lat": 27.6333, "lon": 78.6500, "state": "Uttar Pradesh"},
        "Etawah": {"lat": 26.7833, "lon": 79.0167, "state": "Uttar Pradesh"},
        "Farrukhabad": {"lat": 27.3833, "lon": 79.5833, "state": "Uttar Pradesh"},
        "Fatehpur": {"lat": 25.9333, "lon": 80.8000, "state": "Uttar Pradesh"},
        "Firozabad": {"lat": 27.1500, "lon": 78.4000, "state": "Uttar Pradesh"},
        "Gautam Buddha Nagar": {"lat": 28.4595, "lon": 77.5200, "state": "Uttar Pradesh"},
        "Ghaziabad": {"lat": 28.6692, "lon": 77.4538, "state": "Uttar Pradesh"},
        "Ghazipur": {"lat": 25.5833, "lon": 83.5833, "state": "Uttar Pradesh"},
        "Gonda": {"lat": 27.1333, "lon": 81.9667, "state": "Uttar Pradesh"},
        "Gorakhpur": {"lat": 26.7500, "lon": 83.3667, "state": "Uttar Pradesh"},
        "Hamirpur": {"lat": 25.9500, "lon": 80.1500, "state": "Uttar Pradesh"},
        "Hapur": {"lat": 28.7333, "lon": 77.7667, "state": "Uttar Pradesh"},
        "Hardoi": {"lat": 27.4000, "lon": 80.1333, "state": "Uttar Pradesh"},
        "Hathras": {"lat": 27.6000, "lon": 78.0500, "state": "Uttar Pradesh"},
        "Jalaun": {"lat": 26.1500, "lon": 79.3333, "state": "Uttar Pradesh"},
        "Jaunpur": {"lat": 25.7333, "lon": 82.6833, "state": "Uttar Pradesh"},
        "Jhansi": {"lat": 25.4500, "lon": 78.5667, "state": "Uttar Pradesh"},
        "Kannauj": {"lat": 27.0500, "lon": 79.9167, "state": "Uttar Pradesh"},
        "Kanpur Dehat": {"lat": 26.4667, "lon": 79.9167, "state": "Uttar Pradesh"},
        "Kanpur Nagar": {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh"},
        "Kasganj": {"lat": 27.8000, "lon": 78.6500, "state": "Uttar Pradesh"},
        "Kaushambi": {"lat": 25.5333, "lon": 81.3833, "state": "Uttar Pradesh"},
        "Kheri": {"lat": 28.0333, "lon": 80.7667, "state": "Uttar Pradesh"},
        "Kushinagar": {"lat": 26.7333, "lon": 83.8833, "state": "Uttar Pradesh"},
        "Lalitpur": {"lat": 24.6833, "lon": 78.4167, "state": "Uttar Pradesh"},
        "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
        "Maharajganj": {"lat": 27.1333, "lon": 83.5500, "state": "Uttar Pradesh"},
        "Mahoba": {"lat": 25.2833, "lon": 79.8667, "state": "Uttar Pradesh"},
        "Mainpuri": {"lat": 27.2333, "lon": 79.0167, "state": "Uttar Pradesh"},
        "Mathura": {"lat": 27.5000, "lon": 77.6833, "state": "Uttar Pradesh"},
        "Mau": {"lat": 25.9500, "lon": 83.5500, "state": "Uttar Pradesh"},
        "Meerut": {"lat": 28.9845, "lon": 77.7064, "state": "Uttar Pradesh"},
        "Mirzapur": {"lat": 25.1333, "lon": 82.5667, "state": "Uttar Pradesh"},
        "Moradabad": {"lat": 28.8333, "lon": 78.7667, "state": "Uttar Pradesh"},
        "Muzaffarnagar": {"lat": 29.4667, "lon": 77.7000, "state": "Uttar Pradesh"},
        "Pilibhit": {"lat": 28.6333, "lon": 79.8000, "state": "Uttar Pradesh"},
        "Pratapgarh": {"lat": 25.9000, "lon": 81.9500, "state": "Uttar Pradesh"},
        "Prayagraj": {"lat": 25.4358, "lon": 81.8463, "state": "Uttar Pradesh"},
        "Raebareli": {"lat": 26.2333, "lon": 81.2333, "state": "Uttar Pradesh"},
        "Rampur": {"lat": 28.8000, "lon": 79.0333, "state": "Uttar Pradesh"},
        "Saharanpur": {"lat": 29.9667, "lon": 77.5500, "state": "Uttar Pradesh"},
        "Sambhal": {"lat": 28.5833, "lon": 78.5500, "state": "Uttar Pradesh"},
        "Sant Kabir Nagar": {"lat": 26.7667, "lon": 83.0333, "state": "Uttar Pradesh"},
        "Shahjahanpur": {"lat": 27.8833, "lon": 79.9000, "state": "Uttar Pradesh"},
        "Shamli": {"lat": 29.4500, "lon": 77.3167, "state": "Uttar Pradesh"},
        "Shravasti": {"lat": 27.5167, "lon": 81.7667, "state": "Uttar Pradesh"},
        "Siddharthnagar": {"lat": 27.2833, "lon": 83.1000, "state": "Uttar Pradesh"},
        "Sitapur": {"lat": 27.5667, "lon": 80.6833, "state": "Uttar Pradesh"},
        "Sonbhadra": {"lat": 24.7000, "lon": 83.0667, "state": "Uttar Pradesh"},
        "Sultanpur": {"lat": 26.2667, "lon": 82.0667, "state": "Uttar Pradesh"},
        "Unnao": {"lat": 26.5500, "lon": 80.4833, "state": "Uttar Pradesh"},
        "Varanasi": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh"},
        
        # Uttarakhand - All 13 districts
        "Almora": {"lat": 29.5833, "lon": 79.6500, "state": "Uttarakhand"},
        "Bageshwar": {"lat": 29.8333, "lon": 79.7667, "state": "Uttarakhand"},
        "Chamoli": {"lat": 30.4000, "lon": 79.3167, "state": "Uttarakhand"},
        "Champawat": {"lat": 29.3333, "lon": 80.0833, "state": "Uttarakhand"},
        "Dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
        "Haridwar": {"lat": 29.9457, "lon": 78.1642, "state": "Uttarakhand"},
        "Nainital": {"lat": 29.3803, "lon": 79.4636, "state": "Uttarakhand"},
        "Pauri Garhwal": {"lat": 30.1500, "lon": 78.7833, "state": "Uttarakhand"},
        "Pithoragarh": {"lat": 29.5833, "lon": 80.2167, "state": "Uttarakhand"},
        "Rudraprayag": {"lat": 30.2833, "lon": 78.9833, "state": "Uttarakhand"},
        "Tehri Garhwal": {"lat": 30.3833, "lon": 78.4833, "state": "Uttarakhand"},
        "Udham Singh Nagar": {"lat": 29.0000, "lon": 79.4167, "state": "Uttarakhand"},
        "Uttarkashi": {"lat": 30.7333, "lon": 78.4500, "state": "Uttarakhand"},
        
        # West Bengal - All 23 districts
        "Alipurduar": {"lat": 26.4833, "lon": 89.5333, "state": "West Bengal"},
        "Bankura": {"lat": 23.2333, "lon": 87.0667, "state": "West Bengal"},
        "Birbhum": {"lat": 24.0333, "lon": 87.6167, "state": "West Bengal"},
        "Cooch Behar": {"lat": 26.3167, "lon": 89.4500, "state": "West Bengal"},
        "Dakshin Dinajpur": {"lat": 25.2167, "lon": 88.6500, "state": "West Bengal"},
        "Darjeeling": {"lat": 27.0360, "lon": 88.2627, "state": "West Bengal"},
        "Hooghly": {"lat": 22.9000, "lon": 88.4000, "state": "West Bengal"},
        "Howrah": {"lat": 22.5958, "lon": 88.2636, "state": "West Bengal"},
        "Jalpaiguri": {"lat": 26.5167, "lon": 88.7333, "state": "West Bengal"},
        "Jhargram": {"lat": 22.4500, "lon": 86.9833, "state": "West Bengal"},
        "Kalimpong": {"lat": 27.0667, "lon": 88.4667, "state": "West Bengal"},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
        "Malda": {"lat": 25.0000, "lon": 88.1333, "state": "West Bengal"},
        "Murshidabad": {"lat": 24.1833, "lon": 88.2833, "state": "West Bengal"},
        "Nadia": {"lat": 23.4667, "lon": 88.5667, "state": "West Bengal"},
        "North 24 Parganas": {"lat": 22.6167, "lon": 88.4000, "state": "West Bengal"},
        "Paschim Bardhaman": {"lat": 23.2500, "lon": 87.8500, "state": "West Bengal"},
        "Paschim Medinipur": {"lat": 22.4333, "lon": 87.3167, "state": "West Bengal"},
        "Purba Bardhaman": {"lat": 23.2500, "lon": 87.8500, "state": "West Bengal"},
        "Purba Medinipur": {"lat": 22.0000, "lon": 87.7500, "state": "West Bengal"},
        "Purulia": {"lat": 23.3333, "lon": 86.3667, "state": "West Bengal"},
        "South 24 Parganas": {"lat": 22.1667, "lon": 88.4333, "state": "West Bengal"},
        "Uttar Dinajpur": {"lat": 25.6167, "lon": 88.1333, "state": "West Bengal"},
        
        # Union Territories
        
        # Andaman and Nicobar Islands - All 3 districts
        "Nicobar": {"lat": 7.0333, "lon": 93.7833, "state": "Andaman and Nicobar Islands"},
        "North and Middle Andaman": {"lat": 12.9167, "lon": 92.9167, "state": "Andaman and Nicobar Islands"},
        "South Andaman": {"lat": 11.6667, "lon": 92.7500, "state": "Andaman and Nicobar Islands"},
        
        # Chandigarh - 1 district
        "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Chandigarh"},
        
        # Dadra and Nagar Haveli and Daman and Diu - All 3 districts
        "Dadra and Nagar Haveli": {"lat": 20.2667, "lon": 73.0167, "state": "Dadra and Nagar Haveli and Daman and Diu"},
        "Daman": {"lat": 20.4167, "lon": 72.8333, "state": "Dadra and Nagar Haveli and Daman and Diu"},
        "Diu": {"lat": 20.7167, "lon": 70.9833, "state": "Dadra and Nagar Haveli and Daman and Diu"},
        
        # Delhi - All 11 districts
        "Central Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
        "East Delhi": {"lat": 28.6139, "lon": 77.2850, "state": "Delhi"},
        "New Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
        "North Delhi": {"lat": 28.6692, "lon": 77.2265, "state": "Delhi"},
        "North East Delhi": {"lat": 28.6692, "lon": 77.2850, "state": "Delhi"},
        "North West Delhi": {"lat": 28.6692, "lon": 77.1265, "state": "Delhi"},
        "Shahdara": {"lat": 28.6692, "lon": 77.2850, "state": "Delhi"},
        "South Delhi": {"lat": 28.5355, "lon": 77.2090, "state": "Delhi"},
        "South East Delhi": {"lat": 28.5355, "lon": 77.2850, "state": "Delhi"},
        "South West Delhi": {"lat": 28.5355, "lon": 77.1265, "state": "Delhi"},
        "West Delhi": {"lat": 28.6139, "lon": 77.1265, "state": "Delhi"},
        
        # Jammu and Kashmir - All 20 districts
        "Anantnag": {"lat": 33.7333, "lon": 75.1500, "state": "Jammu and Kashmir"},
        "Bandipora": {"lat": 34.4167, "lon": 74.6333, "state": "Jammu and Kashmir"},
        "Baramulla": {"lat": 34.2000, "lon": 74.3500, "state": "Jammu and Kashmir"},
        "Budgam": {"lat": 34.0167, "lon": 74.7333, "state": "Jammu and Kashmir"},
        "Doda": {"lat": 33.1333, "lon": 75.5500, "state": "Jammu and Kashmir"},
        "Ganderbal": {"lat": 34.2333, "lon": 74.7833, "state": "Jammu and Kashmir"},
        "Jammu": {"lat": 32.7333, "lon": 74.8667, "state": "Jammu and Kashmir"},
        "Kathua": {"lat": 32.3667, "lon": 75.5167, "state": "Jammu and Kashmir"},
        "Kishtwar": {"lat": 33.3167, "lon": 75.7667, "state": "Jammu and Kashmir"},
        "Kulgam": {"lat": 33.6333, "lon": 75.0167, "state": "Jammu and Kashmir"},
        "Kupwara": {"lat": 34.5167, "lon": 74.2500, "state": "Jammu and Kashmir"},
        "Poonch": {"lat": 33.7667, "lon": 74.0833, "state": "Jammu and Kashmir"},
        "Pulwama": {"lat": 33.8667, "lon": 74.8833, "state": "Jammu and Kashmir"},
        "Rajouri": {"lat": 33.3833, "lon": 74.3167, "state": "Jammu and Kashmir"},
        "Ramban": {"lat": 33.2500, "lon": 75.2333, "state": "Jammu and Kashmir"},
        "Reasi": {"lat": 33.0833, "lon": 74.8333, "state": "Jammu and Kashmir"},
        "Samba": {"lat": 32.5667, "lon": 75.1167, "state": "Jammu and Kashmir"},
        "Shopian": {"lat": 33.7167, "lon": 74.8333, "state": "Jammu and Kashmir"},
        "Srinagar": {"lat": 34.0833, "lon": 74.7833, "state": "Jammu and Kashmir"},
        "Udhampur": {"lat": 32.9167, "lon": 75.1333, "state": "Jammu and Kashmir"},
        
        # Ladakh - All 2 districts
        "Kargil": {"lat": 34.5500, "lon": 76.1333, "state": "Ladakh"},
        "Leh": {"lat": 34.1667, "lon": 77.5833, "state": "Ladakh"},
        
        # Lakshadweep - 1 district
        "Lakshadweep": {"lat": 10.5667, "lon": 72.6333, "state": "Lakshadweep"},
        
        # Puducherry - All 4 districts
        "Karaikal": {"lat": 10.9167, "lon": 79.8333, "state": "Puducherry"},
        "Mahe": {"lat": 11.7000, "lon": 75.5333, "state": "Puducherry"},
        "Puducherry": {"lat": 11.9333, "lon": 79.8333, "state": "Puducherry"},
        "Yanam": {"lat": 16.7333, "lon": 82.2167, "state": "Puducherry"}
    }
    
    min_distance = float('inf')
    nearest = None
    
    for district, coords in district_coords.items():
        # Calculate distance using Haversine formula (simplified)
        distance = ((lat - coords['lat'])**2 + (lon - coords['lon'])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            nearest = {
                'district': district,
                'state': coords['state'],
                'distance': distance
            }
    
    return nearest

def get_average_weather_data(state, district, season, soil_type):
    """Get average weather data from our dataset"""
    try:
        # Try to read from the progress file first, then the complete dataset
        dataset_files = [
            'real_india_crop_dataset.csv',
            'real_data_progress_300.csv',
            'complete_india_crop_dataset.csv'
        ]
        
        df = None
        for file in dataset_files:
            try:
                df = pd.read_csv(file)
                break
            except:
                continue
        
        if df is None:
            return None
        
        # Filter data based on state, district, season, and soil type
        filtered_data = df[
            (df['State'] == state) & 
            (df['District'] == district) & 
            (df['Season'] == season)
        ]
        
        # If exact match not found, try with just state and season
        if filtered_data.empty:
            filtered_data = df[
                (df['State'] == state) & 
                (df['Season'] == season)
            ]
        
        # If still no match, use state average
        if filtered_data.empty:
            filtered_data = df[df['State'] == state]
        
        # If still no match, use similar climate zone
        if filtered_data.empty:
            # Determine climate zone based on location
            climate_zone = determine_climate_zone_by_location(state, district)
            filtered_data = df[
                (df['Climate_Zone'] == climate_zone) & 
                (df['Season'] == season)
            ]
        
        if not filtered_data.empty:
            # Get average values
            raw_rainfall = filtered_data['Avg_Rainfall_mm'].mean()
            
            # Scale rainfall to fit the model's expected range (20-300mm)
            # Our dataset has annual values, but model expects seasonal/monthly values
            scaled_rainfall = scale_rainfall_for_model(raw_rainfall, season)
            
            avg_data = {
                'temperature': round(filtered_data['Avg_Temperature_C'].mean(), 1),
                'humidity': round(filtered_data['Avg_Humidity_%'].mean(), 0),
                'rainfall': round(scaled_rainfall, 0),
                'raw_rainfall': round(raw_rainfall, 0),  # Keep original for display
                'climate_zone': filtered_data['Climate_Zone'].iloc[0] if 'Climate_Zone' in filtered_data.columns else 'Unknown'
            }
            return avg_data
        
        return None
        
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return None

def scale_rainfall_for_model(raw_rainfall, season):
    """Scale rainfall values to fit the model's expected range (20-300mm)"""
    
    # Seasonal scaling factors (what portion of annual rainfall occurs in each season)
    seasonal_factors = {
        'Kharif': 0.7,    # 70% of annual rainfall (monsoon season)
        'Rabi': 0.15,     # 15% of annual rainfall (winter season)
        'Zaid': 0.15      # 15% of annual rainfall (summer season)
    }
    
    # Apply seasonal factor
    seasonal_rainfall = raw_rainfall * seasonal_factors.get(season, 0.33)
    
    # Further scale to fit model range (20-300mm)
    # Map the typical range of seasonal rainfall (50-2000mm) to model range (20-300mm)
    if seasonal_rainfall < 50:
        scaled = 20 + (seasonal_rainfall - 20) * 0.5
    elif seasonal_rainfall > 1000:
        scaled = 200 + (seasonal_rainfall - 1000) * 0.1
    else:
        # Linear scaling for the main range
        scaled = 20 + (seasonal_rainfall - 50) * (280 / 950)  # Scale 50-1000 to 20-300
    
    # Ensure it's within bounds
    scaled = max(20, min(300, scaled))
    
    return scaled

def determine_climate_zone_by_location(state, district):
    """Determine climate zone based on state and district"""
    coastal_states = ["Andhra Pradesh", "Gujarat", "Karnataka", "Kerala", "Maharashtra", "Tamil Nadu", "West Bengal"]
    arid_states = ["Rajasthan"]
    tropical_states = ["Assam", "Bihar", "West Bengal"]
    mountain_states = ["Himachal Pradesh", "Uttarakhand"]
    
    if state in coastal_states:
        return "Coastal"
    elif state in arid_states:
        return "Arid"
    elif state in tropical_states:
        return "Tropical"
    elif state in mountain_states:
        return "Mountain"
    else:
        return "Semi-arid"

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template("crop_recommendation.html")

@app.route('/fertiliser-recommendation')
def fertiliser_recommendation():
    return render_template("fertiliser_recommendation.html")

@app.route('/recommend-fertiliser', methods=['POST'])
def recommend_fertiliser():
    try:
        crop = request.form['crop']
        current_n = float(request.form['nitrogen'])
        current_p = float(request.form['phosphorus'])
        current_k = float(request.form['potassium'])
        ph = float(request.form['ph'])
        land_area = float(request.form['land_area'])
        
        # Ideal NPK values for different crops (kg/ha)
        ideal_values = {
            'Rice': {'N': 120, 'P': 60, 'K': 40},
            'Maize': {'N': 150, 'P': 75, 'K': 50},
            'Jute': {'N': 80, 'P': 40, 'K': 40},
            'Cotton': {'N': 120, 'P': 60, 'K': 50},
            'Coconut': {'N': 100, 'P': 40, 'K': 140},
            'Papaya': {'N': 200, 'P': 200, 'K': 400},
            'Orange': {'N': 100, 'P': 50, 'K': 100},
            'Apple': {'N': 100, 'P': 50, 'K': 100},
            'Muskmelon': {'N': 100, 'P': 60, 'K': 80},
            'Watermelon': {'N': 100, 'P': 60, 'K': 80},
            'Grapes': {'N': 100, 'P': 50, 'K': 100},
            'Mango': {'N': 100, 'P': 50, 'K': 100},
            'Banana': {'N': 200, 'P': 60, 'K': 200},
            'Pomegranate': {'N': 100, 'P': 50, 'K': 100},
            'Lentil': {'N': 20, 'P': 60, 'K': 20},
            'Blackgram': {'N': 20, 'P': 60, 'K': 20},
            'Mungbean': {'N': 20, 'P': 60, 'K': 20},
            'Mothbeans': {'N': 20, 'P': 40, 'K': 20},
            'Pigeonpeas': {'N': 20, 'P': 60, 'K': 20},
            'Kidneybeans': {'N': 25, 'P': 60, 'K': 25},
            'Chickpea': {'N': 20, 'P': 60, 'K': 20},
            'Coffee': {'N': 100, 'P': 50, 'K': 100}
        }
        
        ideal = ideal_values.get(crop, {'N': 100, 'P': 50, 'K': 50})
        
        # Detect deficiencies
        deficiencies = []
        n_deficit = max(0, ideal['N'] - current_n)
        p_deficit = max(0, ideal['P'] - current_p)
        k_deficit = max(0, ideal['K'] - current_k)
        
        if n_deficit > 0:
            deficiencies.append({
                'nutrient': 'Nitrogen (N)',
                'current': round(current_n, 2),
                'required': ideal['N'],
                'deficit': round(n_deficit, 2)
            })
        
        if p_deficit > 0:
            deficiencies.append({
                'nutrient': 'Phosphorus (P)',
                'current': round(current_p, 2),
                'required': ideal['P'],
                'deficit': round(p_deficit, 2)
            })
        
        if k_deficit > 0:
            deficiencies.append({
                'nutrient': 'Potassium (K)',
                'current': round(current_k, 2),
                'required': ideal['K'],
                'deficit': round(k_deficit, 2)
            })
        
        # Map to fertilizers and calculate quantities
        fertilizers = []
        
        if n_deficit > 0:
            # Urea contains 46% N
            urea_needed = (n_deficit * land_area * 2.47) / 0.46  # Convert acres to hectares
            fertilizers.append({
                'name': 'Urea',
                'description': 'High nitrogen content fertilizer (46% N)',
                'quantity': round(urea_needed, 2),
                'application': 'Apply in 2-3 splits: at sowing, tillering, and flowering stage',
                'buy_url': 'https://www.google.com/search?q=buy+urea+fertilizer+lowest+price+india&tbm=shop'
            })
        
        if p_deficit > 0:
            # DAP contains 18% N and 46% P2O5
            dap_needed = (p_deficit * land_area * 2.47) / 0.20
            fertilizers.append({
                'name': 'DAP (Di-Ammonium Phosphate)',
                'description': 'Phosphorus-rich fertilizer (46% P2O5, 18% N)',
                'quantity': round(dap_needed, 2),
                'application': 'Apply as basal dose at the time of sowing',
                'buy_url': 'https://www.google.com/search?q=buy+DAP+fertilizer+lowest+price+india&tbm=shop'
            })
        
        if k_deficit > 0:
            # MOP contains 60% K2O
            mop_needed = (k_deficit * land_area * 2.47) / 0.50
            fertilizers.append({
                'name': 'MOP (Muriate of Potash)',
                'description': 'Potassium fertilizer (60% K2O)',
                'quantity': round(mop_needed, 2),
                'application': 'Apply in 2 splits: half at sowing, half at flowering',
                'buy_url': 'https://www.google.com/search?q=buy+MOP+potash+fertilizer+lowest+price+india&tbm=shop'
            })
        
        # pH correction
        if ph < 6.0:
            lime_needed = (6.5 - ph) * 2000 * land_area * 2.47
            fertilizers.append({
                'name': 'Agricultural Lime',
                'description': 'To correct soil acidity',
                'quantity': round(lime_needed, 2),
                'application': 'Apply 2-3 weeks before sowing and mix well with soil',
                'buy_url': 'https://www.google.com/search?q=buy+agricultural+lime+lowest+price+india&tbm=shop'
            })
        elif ph > 7.5:
            gypsum_needed = (ph - 7.0) * 1000 * land_area * 2.47
            fertilizers.append({
                'name': 'Gypsum',
                'description': 'To correct soil alkalinity',
                'quantity': round(gypsum_needed, 2),
                'application': 'Apply and incorporate into soil before sowing',
                'buy_url': 'https://www.google.com/search?q=buy+gypsum+fertilizer+lowest+price+india&tbm=shop'
            })
        
        if not fertilizers:
            fertilizers.append({
                'name': 'No Fertilizer Needed',
                'description': 'Your soil has optimal nutrient levels for this crop',
                'quantity': 0,
                'application': 'Maintain current soil health with organic matter'
            })
        
        recommendation = {
            'crop': crop,
            'land_area': land_area,
            'deficiencies': deficiencies,
            'fertilizers': fertilizers
        }
        
    except Exception as e:
        print(f"Error: {e}")
        recommendation = None
    
    return render_template('fertiliser_recommendation.html', recommendation=recommendation)

@app.route('/scheme-eligibility')
def scheme_eligibility():
    return render_template("scheme_eligibility.html")

@app.route('/check-eligibility', methods=['POST'])
def check_eligibility():
    try:
        # Prepare data for external API in the exact format it expects
        form_data = {
            'state': request.form['state'],
            'district': request.form['district'],
            'land_size': float(request.form['land_size']),
            'crop_type': request.form['crop_type'],
            'category': request.form['category'].upper(),
            'annual_income': float(request.form['annual_income']),
            'income_tax_payer': request.form['income_tax_payer'] == 'Yes',
            'pension': float(request.form.get('pension_amount', 0)) if request.form['receiving_pension'] == 'Yes' else 0,
            'electricity_connection': request.form['electricity_connection'] == 'Yes'
        }
        
        print("Sending data to API:", form_data)
        
        # Call external API with longer timeout for Hugging Face Spaces
        api_url = 'https://sivasai07-niti-setu-eligibility.hf.space/check-eligibility'
        response = requests.post(api_url, json=form_data, timeout=60)  # Increased timeout to 60 seconds
        
        print("API Response Status:", response.status_code)
        print("API Response:", response.text[:1000])
        
        if response.status_code == 200:
            result = response.json()
            print("Parsed result status:", result.get('status'))
            
            if result.get('status') == 'success':
                data = result.get('data', {})
                eligible_schemes = []
                
                print("Available schemes in data:", list(data.keys()))
                
                # Parse PM-KISAN
                if 'PM-KISAN' in data and data['PM-KISAN'].get('eligible'):
                    pm_kisan = data['PM-KISAN']
                    print("PM-KISAN eligible:", pm_kisan)
                    eligible_schemes.append({
                        'name': 'PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)',
                        'description': pm_kisan.get('reason_message', ''),
                        'benefit': pm_kisan.get('benefit_summary', '₹6,000 per year'),
                        'eligibility': f"Confidence: {pm_kisan.get('confidence_score', 0)*100:.0f}%",
                        'next_steps': pm_kisan.get('next_steps', []),
                        'apply_url': 'https://pmkisan.gov.in/'
                    })
                
                # Parse PM-KUSUM
                if 'PM-KUSUM' in data and data['PM-KUSUM'].get('eligible'):
                    pm_kusum = data['PM-KUSUM']
                    print("PM-KUSUM eligible:", pm_kusum)
                    best_component = pm_kusum.get('best_component', 'Component_C')
                    component_data = pm_kusum.get('components', {}).get(best_component, {})
                    
                    eligible_schemes.append({
                        'name': f'PM-KUSUM ({best_component})',
                        'description': component_data.get('reason', pm_kusum.get('reason_message', '')),
                        'benefit': f"{component_data.get('subsidy_percent', pm_kusum.get('subsidy_percent', 60))}% subsidy on solar infrastructure",
                        'eligibility': f"Confidence: {pm_kusum.get('confidence_score', 0)*100:.0f}%",
                        'next_steps': pm_kusum.get('next_steps', []),
                        'apply_url': 'https://pmkusum.mnre.gov.in/'
                    })
                
                # Parse AIF
                if 'Agriculture Infrastructure Fund (AIF)' in data and data['Agriculture Infrastructure Fund (AIF)'].get('eligible'):
                    aif = data['Agriculture Infrastructure Fund (AIF)']
                    print("AIF eligible:", aif)
                    eligible_schemes.append({
                        'name': 'Agriculture Infrastructure Fund (AIF)',
                        'description': aif.get('reason_message', ''),
                        'benefit': aif.get('benefit_summary', '3% interest subvention'),
                        'eligibility': f"Confidence: {aif.get('confidence_score', 0)*100:.0f}%",
                        'next_steps': aif.get('next_steps', []),
                        'apply_url': 'https://agriinfra.dac.gov.in/'
                    })
                
                print(f"Found {len(eligible_schemes)} eligible schemes")
            else:
                print("API returned unsuccessful status")
                eligible_schemes = []
        else:
            print("API returned non-200 status")
            eligible_schemes = []
            
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        eligible_schemes = []
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        eligible_schemes = []
    
    return render_template('scheme_eligibility.html', schemes=eligible_schemes)

@app.route("/predict-crop", methods=['POST'])
def predict_crop():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled = crop_ms.transform(single_pred)
        final_features = crop_sc.transform(scaled)
        
        probabilities = crop_model.predict_proba(final_features)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Crop information database
        crop_info = {
            "Rice": {"season": "Kharif (June-Nov)", "duration": "3-6 months", "water": "High", "soil": "Clay/Loamy", "description": "Rice is a staple food crop requiring flooded conditions and warm climate."},
            "Maize": {"season": "Kharif/Rabi", "duration": "3-4 months", "water": "Moderate", "soil": "Well-drained loamy", "description": "Maize is versatile and can be grown in various seasons with moderate water needs."},
            "Jute": {"season": "Kharif (April-June)", "duration": "4-5 months", "water": "High", "soil": "Alluvial", "description": "Jute is a fiber crop that thrives in warm, humid conditions."},
            "Cotton": {"season": "Kharif (May-June)", "duration": "5-6 months", "water": "Moderate", "soil": "Black/Alluvial", "description": "Cotton requires warm weather and moderate rainfall for optimal growth."},
            "Coconut": {"season": "Year-round", "duration": "Perennial", "water": "High", "soil": "Sandy loam", "description": "Coconut is a tropical crop requiring consistent moisture and warm climate."},
            "Papaya": {"season": "Year-round", "duration": "9-12 months", "water": "Moderate", "soil": "Well-drained loamy", "description": "Papaya grows best in tropical and subtropical regions."},
            "Orange": {"season": "Year-round", "duration": "Perennial", "water": "Moderate", "soil": "Well-drained", "description": "Orange trees require warm climate and well-distributed rainfall."},
            "Apple": {"season": "Temperate", "duration": "Perennial", "water": "Moderate", "soil": "Well-drained loamy", "description": "Apples grow best in cool temperate regions with cold winters."},
            "Muskmelon": {"season": "Summer", "duration": "2-3 months", "water": "Moderate", "soil": "Sandy loam", "description": "Muskmelon is a warm-season crop requiring good drainage."},
            "Watermelon": {"season": "Summer", "duration": "2-3 months", "water": "Moderate", "soil": "Sandy loam", "description": "Watermelon thrives in warm weather with adequate moisture."},
            "Grapes": {"season": "Year-round", "duration": "Perennial", "water": "Moderate", "soil": "Well-drained", "description": "Grapes require warm, dry summers and mild winters."},
            "Mango": {"season": "Year-round", "duration": "Perennial", "water": "Moderate", "soil": "Well-drained", "description": "Mango is a tropical fruit tree requiring warm climate."},
            "Banana": {"season": "Year-round", "duration": "9-12 months", "water": "High", "soil": "Rich loamy", "description": "Banana requires warm, humid conditions and rich soil."},
            "Pomegranate": {"season": "Year-round", "duration": "Perennial", "water": "Low-Moderate", "soil": "Well-drained", "description": "Pomegranate is drought-tolerant and grows in arid regions."},
            "Lentil": {"season": "Rabi (Oct-Nov)", "duration": "3-4 months", "water": "Low", "soil": "Loamy", "description": "Lentil is a cool-season pulse crop with low water needs."},
            "Blackgram": {"season": "Kharif/Rabi", "duration": "2-3 months", "water": "Low-Moderate", "soil": "Loamy", "description": "Blackgram is a short-duration pulse crop."},
            "Mungbean": {"season": "Kharif/Summer", "duration": "2-3 months", "water": "Low-Moderate", "soil": "Loamy", "description": "Mungbean is a quick-growing pulse crop."},
            "Mothbeans": {"season": "Kharif", "duration": "2-3 months", "water": "Low", "soil": "Sandy", "description": "Mothbeans are drought-resistant and grow in arid regions."},
            "Pigeonpeas": {"season": "Kharif", "duration": "5-6 months", "water": "Low-Moderate", "soil": "Well-drained", "description": "Pigeonpeas are drought-tolerant pulse crops."},
            "Kidneybeans": {"season": "Kharif/Rabi", "duration": "3-4 months", "water": "Moderate", "soil": "Well-drained loamy", "description": "Kidneybeans require moderate water and well-drained soil."},
            "Chickpea": {"season": "Rabi (Oct-Nov)", "duration": "4-5 months", "water": "Low-Moderate", "soil": "Loamy", "description": "Chickpea is a cool-season pulse crop."},
            "Coffee": {"season": "Year-round", "duration": "Perennial", "water": "High", "soil": "Well-drained loamy", "description": "Coffee requires high rainfall and shade in tropical regions."}
        }
        
        crops = []
        for idx in top_3_indices:
            crop_name = crop_dict[idx + 1]
            info = crop_info.get(crop_name, {
                "season": "N/A", "duration": "N/A", "water": "N/A", 
                "soil": "N/A", "description": "Information not available."
            })
            crops.append({
                'name': crop_name,
                'probability': round(probabilities[idx] * 100, 1),
                'season': info['season'],
                'duration': info['duration'],
                'water': info['water'],
                'description': info['description']
            })
            
    except Exception as e:
        crops = None
        
    return render_template('crop_recommendation.html', crops=crops)

if __name__ == "__main__":
    app.run(debug=True)
        
