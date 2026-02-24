import requests
import json

# Your API key from pokemontcg.io
API_KEY = "11d95695-4140-4823-bde7-f5a7ac40ec93" # this should never be shown publically
BASE_URL = "https://api.pokemontcg.io/v2/cards"

headers = {
    "X-Api-Key": API_KEY # the header becomes the api key
}

def search_pokemon_card(card_name):
    params = {
        "q": f'name:"{card_name}"',
        "pageSize": 1  # Only get the first match for now
    }
    response = requests.get(BASE_URL, headers=headers, params=params)
    response.raise_for_status()  # Raise error if request fails
    data = response.json().get("data", [])

    if not data: # if no card data is returned 
        print("No card found.")
        return

    card = data[0]  # First result
    #print(json.dumps(data, indent=2))
    # print(f"Price: {card['prices'].get('normal', {}).get('low', 'N/A')}") #how to get access to the price 
    print(f"Price low: {card['tcgplayer']['prices'].get('normal', {}).get('low', 'N/A')}")
    print(f"Price mid: {card['tcgplayer']['prices'].get('normal', {}).get('mid', 'N/A')}")
    print(f"Price high: {card['tcgplayer']['prices'].get('normal', {}).get('high', 'N/A')}")
    print(f"Price market: {card['tcgplayer']['prices'].get('normal', {}).get('market', 'N/A')}")
    #print(f"Name: {card['name']}")
    #print(f"Set: {card['set']['name']}")
    #print(f"Rarity: {card.get('rarity', 'Unknown')}")
    #print(f"Image URL: {card['images']['small']}")
    #print("-" * 40)

if __name__ == "__main__":
    pokemon_name = input("Enter a Pokémon card name: ")
    search_pokemon_card(pokemon_name)