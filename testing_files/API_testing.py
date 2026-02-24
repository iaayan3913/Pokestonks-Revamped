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
    if response == 504: 
        print("error 504, issue with server")
    response.raise_for_status()  # Raise error if request fails
    data = response.json().get("data", [])

    if not data: # if no card data is returned 
        print("No card found.")
        return

    card = data[0]  # First result
    #print(data)
    #print(f"Price: {card['prices'].get('normal', {}).get('low', 'N/A')}") #how to get access to the price 
    print(f"Price: {card['prices'].get('1stEditiionHolofoil', {}).get('low', 'N/A')}") #how to get access to the price
    print(f"Name: {card['name']}")
    print(f"Set: {card['set']['name']}")
    print(f"Rarity: {card.get('rarity', 'Unknown')}")
    print(f"Image URL: {card['images']['small']}")
    print("-" * 40)

if __name__ == "__main__":
    pokemon_name = input("Enter a Pokémon card name: ")
    search_pokemon_card(pokemon_name)


# what needs to be done: 
# Abilities, hp, energy costs, damage output, evolutions are what i think of
# ensure no duplicates by checking with ID
# to refine the search get users to input name/ id 


# pokeprice_free_e3a56d7189c4db088c94e7674f9800ab3af45d1c6b27952a