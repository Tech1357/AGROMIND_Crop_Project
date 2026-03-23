# 🌾 Smart Agriculture Assistant

A complete farming helper app that helps farmers make better decisions about crops, fertilizers, and government schemes.

## What This App Does

### 🌱 **Crop Recommendation**
- Tells you which crops to grow based on your soil and weather
- Uses your location to get real weather data
- Shows top 3 best crops with pictures and details
- Works for all districts in India

### 🧪 **Fertilizer Recommendation** 
- Suggests which fertilizers your crops need
- Calculates exact amounts of NPK (Nitrogen, Phosphorus, Potassium)
- Helps fix soil pH problems
- Shows where to buy fertilizers online

### 📋 **Government Scheme Checker**
- Checks which government schemes you can apply for
- Covers PM-KISAN, PM-KUSUM, and Agriculture Infrastructure Fund
- Shows benefits and how to apply
- Works for all Indian states and districts

## How to Use

### 1. **Start the App**
```bash
python app.py
```
Then open your web browser and go to `http://localhost:5000`

### 2. **For Crop Recommendations:**
- Enter your soil test values (from Soil Health Card)
- Click "Get Weather Data for My Location" 
- Choose your season and soil type
- Get crop suggestions with pictures

### 3. **For Fertilizer Help:**
- Enter your soil NPK values and pH
- Select your crop
- Get fertilizer recommendations and buying links

### 4. **For Government Schemes:**
- Fill in your details (state, district, land size, etc.)
- Click "Check Eligibility"
- See which schemes you qualify for

## What You Need

### **Soil Information** (from Soil Health Card):
- Nitrogen (N): 0-140 kg/ha
- Phosphorus (P): 5-145 kg/ha  
- Potassium (K): 5-205 kg/ha
- pH level: 3.5-9.5

### **Other Details:**
- Your location (for weather data)
- Crop type you want to grow
- Land size and income (for schemes)

## Features

✅ **Smart & Easy**: Uses AI to give you the best advice  
✅ **Real Weather**: Gets actual weather data for your area  
✅ **All India Coverage**: Works for 700+ districts across India  
✅ **Government Schemes**: Checks eligibility for farmer schemes  
✅ **Fertilizer Calculator**: Tells you exactly what to buy  
✅ **Mobile Friendly**: Works on phones and computers  
✅ **Free to Use**: No cost, no registration needed  

## Installation

1. **Download the code:**
```bash
git clone https://github.com/Tech1357/crop_feb.git
cd crop_feb
```

2. **Install requirements:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
python app.py
```

## Files in This Project

- `app.py` - Main application file
- `templates/` - Web pages
- `static/` - Images and styling
- `model.pkl` - AI model for crop prediction
- `real_india_crop_dataset.csv` - Real weather data for India
- `generate_real_data.py` - Creates weather data

## Supported Crops

Rice, Maize, Cotton, Wheat, Sugarcane, Banana, Mango, Grapes, Apple, Orange, Coconut, Papaya, Watermelon, Muskmelon, Pomegranate, Lentil, Chickpea, Blackgram, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Coffee, Jute

## Supported Areas

All 28 states and 8 union territories of India with 700+ districts including:
- Andhra Pradesh, Telangana, Karnataka, Tamil Nadu
- Maharashtra, Gujarat, Rajasthan, Madhya Pradesh  
- Uttar Pradesh, Bihar, West Bengal, Odisha
- Punjab, Haryana, Himachal Pradesh, Uttarakhand
- And all other Indian states and UTs

## Government Schemes Covered

1. **PM-KISAN**: ₹6,000 per year for farmers
2. **PM-KUSUM**: Solar pump subsidies up to 60%
3. **Agriculture Infrastructure Fund**: Low-interest loans

## Need Help?

- Make sure you have your Soil Health Card for accurate results
- The first time you check schemes, it may take 1 minute (the AI is starting up)
- All weather data comes from NASA satellites for accuracy
- Fertilizer prices are from Google Shopping for best deals

## Technical Details

- **Built with**: Python Flask, HTML, CSS, JavaScript
- **AI Model**: Machine Learning for crop prediction  
- **Weather Data**: NASA POWER API and OpenWeatherMap
- **Government Schemes**: RAG AI system with official documents
- **Coverage**: 700+ districts with real coordinates

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Dataset source: Agricultural research data
- Built with Flask web framework
- UI design inspired by modern agricultural technology interfaces

---

**Made for Indian farmers to make better farming decisions! 🇮🇳**