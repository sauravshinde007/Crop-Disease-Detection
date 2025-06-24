import requests

# Replace with your OpenWeather API Key
API_KEY = "6ba9f10d7e0824fe03f3957e901fe614"

# Replace with a valid city name
CITY = "Pune"

# OpenWeather API URL
url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

try:
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ OpenWeather API is working!")
        print(f"🌡 Temperature: {data['main']['temp']}°C")
        print(f"💧 Humidity: {data['main']['humidity']}%")
        print(f"⛅ Condition: {data['weather'][0]['description']}")
    else:
        print(f"❌ API Error: {response.status_code} - {response.json()}")  # Print API error details

except Exception as e:
    print(f"❌ Exception occurred: {str(e)}")
