import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from misc_utils import PokemonEntity, BasicPkmnLogic, load_pokemon
from pathlib import Path

def update_used_moves(pkmn, move):
    if move in pkmn.used_moves:
        pkmn.used_moves[move] += 1
    else:
        pkmn.used_moves[move] = 1

def battle_pokemons(pokemon1: PokemonEntity, pokemon2: PokemonEntity):
    battle_logic = BasicPkmnLogic()
    battle_logic.manual_battle_create(pokemon1, pokemon2)
    
    while not battle_logic.manual_battle.ended:
        if battle_logic.manual_battle.turn > 500:
            break
        choice1 = pokemon1.ask_for_choice(battle_logic.manual_battle.p1.active_pokemon[0], 
                                          battle_logic.manual_battle.p2.active_pokemon[0])
        battle_logic.manual_battle_do_turn(pokemon1, pokemon2, choice1)
        
        # Actualizar movimientos usados
        if 0 <= choice1 < 4:
            move = pokemon1.entity[0].moves[choice1]
            update_used_moves(pokemon1, move)
    
    winner = pokemon1 if battle_logic.manual_battle.winner == 'p1' else pokemon2
    loser = pokemon2 if battle_logic.manual_battle.winner == 'p1' else pokemon1
    
    return winner, loser

def tournament(pokemons):
    for pok in pokemons:
        print("s")
        for _ in range(23):
            foe = np.random.choice(pokemons)
            winner, loser = battle_pokemons(pok, foe)
            winner.won_battles += 1
            winner.total_battles += 1
            loser.total_battles += 1

def analyze_experiment(pokemons, experiment_name):
    # [print(p.used_moves) for p in pokemons]
    # exit()
    move_usage_data = np.array([len(p.used_moves) for p in pokemons])
    print(move_usage_data)
    # Calcular la distribución observada
    observed_distribution = np.bincount(move_usage_data, minlength=5) / len(pokemons)
    
    # Crear una distribución uniforme esperada
    expected_distribution = np.ones(5) / 5
    
    # Realizar la prueba de Kolmogorov-Smirnov
    ks_statistic, p_value = stats.kstest(observed_distribution, stats.uniform(loc=0, scale=5).cdf)
    
    # Calcular la entropía de Shannon
    entropy = stats.entropy(observed_distribution)
    
    # Calcular estadísticas descriptivas
    mean_moves = np.mean(move_usage_data)
    var_moves = np.var(move_usage_data)
    
    print(f"Resultados para {experiment_name}")
    print(f"D de Kolmogorov-Smirnov: {ks_statistic:.4f}")
    print(f"p-valor: {p_value:.4f}")
    print(f"Entropía de Shannon: {entropy:.4f}")
    print(f"Promedio de movimientos usados: {mean_moves:.2f}")
    print(f"Varianza de movimientos usados: {var_moves:.2f}")
    
    # Visualización
    plt.figure(figsize=(10, 6))
    plt.bar(range(5), observed_distribution, alpha=0.8, label='Observado')
    plt.plot(range(5), expected_distribution, 'r--', label='Esperado (Uniforme)')
    plt.title(f'Distribución de Uso de Movimientos - {experiment_name}')
    plt.xlabel('Número de Movimientos Usados')
    plt.ylabel('Frecuencia Relativa')
    plt.legend()
    plt.savefig(f'./experiment_ii/{experiment_name.lower().replace(" ", "_")}_distribution.png')
    plt.close()
    
    return ks_statistic, p_value, entropy, mean_moves, var_moves

def main():
    # Crear una lista de Pokémon (asumiendo que tienes una función para crear Pokémon)
    pokemons = []  # Crear 50 Pokémon para el experimento
    for pat in Path("./experiment_ii/pkmns/").iterdir():
        pk = load_pokemon(pat)
        pk.lvl = 100
        pk.entity[0].level = 100
        pk.won_battles = 0
        pk.total_battles = 0
        pk.used_moves = {}
        pokemons.append(pk)
        
    # Realizar el torneo
    tournament(pokemons)
    
    # Analizar los resultados
    results = analyze_experiment(pokemons, "Experimento de Uso de Movimientos")
    
    # Guardar los datos de cada Pokémon
    for i, pokemon in enumerate(pokemons):
        with open(f'./experiment_ii/results/pokemon_data_{i}.txt', 'w') as f:
            f.write(f"Especie: {pokemon.entity[0].species}\n")
            f.write(f"Batallas ganadas: {pokemon.won_battles}/{pokemon.total_battles}\n")
            f.write(f"Nivel: {pokemon.lvl}\n")
            f.write("Movimientos usados:\n")
            for move, count in pokemon.used_moves.items():
                f.write(f"  {move}: {count} veces\n")
            f.write(f"Total de movimientos diferentes usados: {len(pokemon.used_moves)}\n")

if __name__ == "__main__":
    main()
