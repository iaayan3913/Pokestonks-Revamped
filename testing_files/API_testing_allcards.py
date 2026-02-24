import requests


API_KEY = "11d95695-4140-4823-bde7-f5a7ac40ec93"
BASE_URL = "https://api.pokemontcg.io/v2/cards"
headers = {"X-Api-Key": API_KEY}

def fetch_all_cards(card_name):
    page = 1
    page_size = 50
    all_cards = []

    while True:
        params = {
            "q": f'name:"{card_name}"',
            "page": page,
            "pageSize": page_size
        }
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        cards = data.get("data", [])
        all_cards.extend(cards)

        total_count = data.get("totalCount", 0)
        if page * page_size >= total_count:
            break
        page += 1

    return all_cards

def remove_duplicates_by_id(cards):
    seen_names = set()
    unique_cards = []
    
    for card in cards:
        id = card['id']
        if id not in seen_names:
            unique_cards.append(card)
            seen_names.add(id)
    
    return unique_cards

def print_cards(cards):
    for i, card in enumerate(cards, 1):
        print(f"Card #{i}")
        print(f"ID: {card['id']}")
        print(f"Name: {card['name']}")
        print(f"Set: {card['set']['name']}")
        print(f"Rarity: {card.get('rarity', 'Unknown')}")
        print(f"Release Date: {card['set'].get('releaseDate', 'Unknown')}")
        print(f"Image URL: {card['images']['small']}")
        print("-" * 50)

if __name__ == "__main__":
    name = input("Enter Pokémon card name: ")
    all_cards = fetch_all_cards(name)
    print(f"Fetched {len(all_cards)} cards (including duplicates or variations).")

    unique_cards = remove_duplicates_by_id(all_cards)
    print(f"{len(unique_cards)} cards after removing duplicates by name.\n")

    print_cards(unique_cards)
    
    
    #from this stage, develop a website that shows all of this information
    # allow users to search by name to show multiple cards, or by id to show a specific card
