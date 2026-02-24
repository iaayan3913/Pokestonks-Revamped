import requests
import json

def debug_card_api(api_key, card_id):
    """Debug version with detailed error reporting"""
    url = "https://www.pokemonpricetracker.com/api/prices"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"id": card_id}
    
    print("=== API DEBUG INFO ===")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    print("="*30)
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"Error Status: {response.status_code}")
            print(f"Error Text: {response.text}")
            return None
        
        # Try to parse JSON
        try:
            results = response.json()
            print("JSON Response Structure:")
            print(json.dumps(results, indent=2))
            
            # Check if it's the expected structure
            if isinstance(results, dict):
                if 'data' in results:
                    cards = results['data']
                    print(f"Found {len(cards)} cards in 'data' field")
                elif 'results' in results:
                    cards = results['results']
                    print(f"Found {len(cards)} cards in 'results' field")
                else:
                    print("Available keys in response:", list(results.keys()))
                    cards = results
                    
            elif isinstance(results, list):
                print(f"Response is a list with {len(results)} items")
                cards = results
            else:
                print(f"Unexpected response type: {type(results)}")
                return None
                
            return results
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None

def get_card_by_id_fixed(api_key, card_id):
    """Fixed version based on actual API response structure"""
    url = "https://www.pokemonpricetracker.com/api/prices"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"id": card_id}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        results = response.json()
        
        # Handle different possible response structures
        cards = None
        if isinstance(results, dict):
            if 'data' in results:
                cards = results['data']
            elif 'results' in results:
                cards = results['results']
            elif 'cards' in results:
                cards = results['cards']
            else:
                # Maybe the card data is directly in the response
                cards = [results] if 'name' in results else None
        elif isinstance(results, list):
            cards = results
            
        if cards and len(cards) > 0:
            card_data = cards[0]
            
            # Try different ways to access card info
            name = card_data.get('name', 'Unknown Name')
            
            # Try different price fields
            price = None
            price_fields = ['price', 'current_price', 'market_price', 'avg_price']
            for field in price_fields:
                if field in card_data:
                    price = card_data[field]
                    break
            
            # Try different set info
            set_name = 'Unknown Set'
            if 'set' in card_data:
                if isinstance(card_data['set'], dict):
                    set_name = card_data['set'].get('name', 'Unknown Set')
                else:
                    set_name = str(card_data['set'])
            elif 'set_name' in card_data:
                set_name = card_data['set_name']
                
            print(f"Found card: {name} ({set_name})")
            if price is not None:
                print(f"Current price: ${price}")
            else:
                print("Price information not found")
                print("Available fields:", list(card_data.keys()))
            
            return card_data
        else:
            print(f"No card found with ID: {card_id}")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def try_different_endpoints(api_key, card_id):
    """Try different endpoint variations"""
    endpoints = [
        "https://www.pokemonpricetracker.com/api/prices",
        "https://www.pokemonpricetracker.com/api/cards",
        "https://www.pokemonpricetracker.com/api/card",
        f"https://www.pokemonpricetracker.com/api/prices/{card_id}",
        f"https://www.pokemonpricetracker.com/api/cards/{card_id}"
    ]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for url in endpoints:
        print(f"\nTrying endpoint: {url}")
        try:
            if card_id in url:
                response = requests.get(url, headers=headers)
            else:
                response = requests.get(url, headers=headers, params={"id": card_id})
                
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("SUCCESS! This endpoint works.")
                try:
                    data = response.json()
                    print("Response keys:", list(data.keys()) if isinstance(data, dict) else f"List with {len(data)} items")
                except:
                    print("Response is not JSON")
            else:
                print(f"Failed: {response.text[:200]}")
                
        except Exception as e:
            print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    api_key = "pokeprice_free_e3a56d7189c4db088c94e7674f9800ab3af45d1c6b27952a"  # Add your API key here
    card_id = "svp-48"
    
    print("=== DEBUGGING API CALL ===")
    result = debug_card_api(api_key, card_id)
    
    print("\n=== TRYING FIXED VERSION ===")
    card = get_card_by_id_fixed(api_key, card_id)
    
    print("\n=== TRYING DIFFERENT ENDPOINTS ===")
    try_different_endpoints(api_key, card_id)
    
    # If you want to see what the actual response structure looks like
    if result:
        print("\n=== FULL RESPONSE ANALYSIS ===")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"Key '{key}': {type(value)}")
                if isinstance(value, list) and value:
                    print(f"  - First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else 'Not a dict'}")
                elif isinstance(value, dict):
                    print(f"  - Nested keys: {list(value.keys())}")