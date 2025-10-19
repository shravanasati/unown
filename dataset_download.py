from concurrent.futures import ThreadPoolExecutor
import pypokedex
import csv

MAX_DEX = 1025

# Generation ranges (dex number ranges for each generation)
GENERATION_RANGES = {
    1: (1, 151),  # Gen 1: Red/Blue/Yellow
    2: (152, 251),  # Gen 2: Gold/Silver/Crystal
    3: (252, 386),  # Gen 3: Ruby/Sapphire/Emerald
    4: (387, 493),  # Gen 4: Diamond/Pearl/Platinum
    5: (494, 649),  # Gen 5: Black/White
    6: (650, 721),  # Gen 6: X/Y
    7: (722, 809),  # Gen 7: Sun/Moon
    8: (810, 905),  # Gen 8: Sword/Shield
    9: (906, 1025),  # Gen 9: Scarlet/Violet
}

# Legendary Pokémon dex numbers (from Bulbapedia)
# Total: 71 Legendary Pokémon as of Gen IX
LEGENDARY_POKEMON = {
    # Generation I (4 total)
    144,
    145,
    146,  # Articuno, Zapdos, Moltres
    150,  # Mewtwo
    # Generation II (5 total: 9 cumulative)
    243,
    244,
    245,  # Raikou, Entei, Suicune
    249,
    250,  # Lugia, Ho-Oh
    # Generation III (8 total: 17 cumulative)
    377,
    378,
    379,  # Regirock, Regice, Registeel
    380,
    381,  # Latias, Latios
    382,
    383,
    384,  # Kyogre, Groudon, Rayquaza
    # Generation IV (9 total: 26 cumulative)
    480,
    481,
    482,  # Uxie, Mesprit, Azelf
    483,
    484,
    485,  # Dialga, Palkia, Heatran
    486,
    487,
    488,  # Regigigas, Giratina, Cresselia
    # Generation V (9 total: 35 cumulative)
    638,
    639,
    640,  # Cobalion, Terrakion, Virizion
    641,
    642,
    643,  # Tornadus, Thundurus, Landorus
    644,
    645,
    646,  # Reshiram, Zekrom, Kyurem
    # Generation VI (3 total: 38 cumulative)
    716,
    717,
    718,  # Xerneas, Yveltal, Zygarde
    # Generation VII (11 total: 49 cumulative)
    792,
    793,  # Type: Null, Silvally
    785,
    786,
    787,
    788,  # Tapu Koko, Tapu Lele, Tapu Bulu, Tapu Fini
    789,
    790,  # Cosmog, Cosmoem
    791,
    792,
    793,  # Solgaleo, Lunala, Necrozma
    # Generation VIII (11 total: 60 cumulative)
    # Galarian forms of Gen I birds (no separate dex numbers)
    888,
    889,  # Zacian, Zamazenta
    890,  # Eternatus
    891,
    892,  # Kubfu, Urshifu
    894,
    895,  # Regieleki, Regidrago
    898,
    899,
    900,  # Calyrex, Glastrier, Spectrier
    905,  # Enamorus
    # Generation IX (11 total: 71 cumulative)
    984,
    985,
    986,
    987,  # Wo-Chien, Chien-Pao, Ting-Lu, Chi-Yu
    981,
    982,  # Koraidon, Miraidon
    1000,
    1001,
    1002,  # Okidogi, Munkidori, Fezandipiti
    1003,  # Ogerpon
    1004,  # Terapagos
}


def get_pokemon(dex: int) -> pypokedex.Pokemon:
    return pypokedex.get(dex=dex)


def get_generation(dex: int) -> int:
    """Get the generation number for a Pokémon by dex number."""
    for gen, (start, end) in GENERATION_RANGES.items():
        if start <= dex <= end:
            return gen
    return 0  # Unknown generation


def is_legendary(dex: int) -> bool:
    """Check if a Pokémon is legendary."""
    return dex in LEGENDARY_POKEMON


def extract_pokemon_data(pokemon: pypokedex.Pokemon) -> dict:
    """Extract relevant data from a Pokemon object."""
    stats = pokemon.base_stats
    abilities_list = (
        [ability.name for ability in pokemon.abilities] if pokemon.abilities else []
    )
    base_total = (
        stats.hp
        + stats.attack
        + stats.defense
        + stats.sp_atk
        + stats.sp_def
        + stats.speed
    )
    dex = pokemon.dex
    generation = get_generation(dex)
    legendary = is_legendary(dex)

    return {
        "dex": dex,
        "name": pokemon.name,
        "generation": generation,
        "types": ", ".join(pokemon.types) if pokemon.types else "",
        "abilities": "; ".join(abilities_list),
        "hp": stats.hp,
        "attack": stats.attack,
        "defense": stats.defense,
        "sp_atk": stats.sp_atk,
        "sp_def": stats.sp_def,
        "speed": stats.speed,
        "base_total": base_total,
        "is_legendary": legendary,
        "height": pokemon.height,
        "weight": pokemon.weight,
        "base_experience": pokemon.base_experience,
    }


with ThreadPoolExecutor(max_workers=12) as pool:
    # Submit all download tasks
    futures = [pool.submit(get_pokemon, dex) for dex in range(1, MAX_DEX + 1)]

    # Collect results and write to CSV
    pokemon_data = []
    for i, future in enumerate(futures, 1):
        try:
            pokemon = future.result()
            data = extract_pokemon_data(pokemon)
            pokemon_data.append(data)
            print(f"Downloaded {i}/{MAX_DEX}: {pokemon.name}")
        except Exception as e:
            print(f"Error downloading Pokemon {i}: {e}")

    # Write to CSV file
    if pokemon_data:
        csv_file = "pokemon_data.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pokemon_data[0].keys())
            writer.writeheader()
            writer.writerows(pokemon_data)
        print(f"\nSuccessfully saved {len(pokemon_data)} Pokémon to {csv_file}")
