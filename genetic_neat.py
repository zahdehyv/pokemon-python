
from pokemon import PokemonEntity, load_pokemon, save_pokemon
from pkmn_logic_utils import BasicPkmnLogic, get_U
import numpy as np
import neat
from pathlib import Path

NO_OF_GENS = None

CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn')

class GeneticEvolution:
    def __init__(self):
        self.index_generator = None
        self.genome_id_to_pokemon = {}
        self.pkmn_logic = BasicPkmnLogic()
        self.available_pokemons = []
        self.ax_method = None
        self.m_fitness = 0.0
        self.age = 0
        
    def mutate(self, pkmn: PokemonEntity):
        mutates = 0
        tc = 0
        while get_U() < pkmn.temperature():
            tc = tc +1
            if tc > 7:
                break
            U2 = get_U()
            if U2 < 0.5 and len(pkmn.entity[0].moves)>0 and len(pkmn.other_moves)>0: # mutate move
                index_out = np.random.randint(len(pkmn.entity[0].moves))
                move_out = pkmn.entity[0].moves[index_out]
                index_in = np.random.randint(len(pkmn.other_moves))
                move_in = pkmn.other_moves[index_in]
                pkmn.entity[0].moves[index_out] = move_in
                pkmn.other_moves[index_in] = move_out
                
            elif U2<0.9 and len(pkmn.other_abilities) > 0: #mutate ability
                ability_out = pkmn.entity[0].ability
                ability_in_index = np.random.randint(len(pkmn.other_abilities))
                ability_in = pkmn.other_abilities[ability_in_index]
                pkmn.other_abilities[ability_in_index]= ability_out
                pkmn.entity[0].ability = ability_in
                
            else: #mutate nature
                pass
               
        return mutates
        
    def crossover(self, pkmn_x: PokemonEntity, pkmn_y: PokemonEntity):
        pkmn_x.other_abilities = [pkmn_x.entity[0].ability]+pkmn_x.other_abilities
        pkmn_x.entity[0].ability = np.random.choice([pkmn_x.entity[0].ability, pkmn_y.entity[0].ability])
        if pkmn_x.entity[0].ability in pkmn_x.other_abilities:
            pkmn_x.other_abilities.remove(pkmn_x.entity[0].ability)
        
        pkmn_x.other_natures = [pkmn_x.entity[0].nature]+pkmn_x.other_natures
        pkmn_x.entity[0].nature = np.random.choice([pkmn_x.entity[0].nature, pkmn_y.entity[0].nature])
        if pkmn_x.entity[0].nature in pkmn_x.other_natures:
            pkmn_x.other_natures.remove(pkmn_x.entity[0].nature)
        
        while True:
            moves_universe = set(pkmn_x.other_moves + pkmn_x.entity[0].moves)
            new_moves = []
            for x_mv, y_mv in zip(pkmn_x.entity[0].moves, pkmn_y.entity[0].moves):
                mv = np.random.choice([x_mv, y_mv])
                if mv in moves_universe:
                    new_moves.append(mv)
                    moves_universe.remove(mv)
            if(len(new_moves) == 4):
                pkmn_x.other_moves = list(moves_universe)
                pkmn_x.entity[0].moves = new_moves
                break
            else:
                if get_U()<0.1:
                    break
    

if __name__=="__main__":
    genetic = GeneticEvolution()

    def eval(genomes, config):
        pokemons = []
        mut_n = 0
        newborn_n = 0
        created_n = 0
        selected_n = 0
        genetic.age = genetic.age +1
        for genome_id, genome in genomes:
            pok = None
            if genome_id in genetic.genome_id_to_pokemon:
                pok : PokemonEntity = genetic.genome_id_to_pokemon[genome_id]
                pok.lvl = 1
                pok.entity[0].level = 1
                pok.xp = 0
                pok.genome = genome
                pok.set_ai()
                pokemons.append(pok)
                selected_n = selected_n +1
            else:
                if (genome.selected_parent[0] and genome.selected_parent[1]) and (genome.selected_parent[0] in genetic.genome_id_to_pokemon) and (get_U()<0.7):
                    parent: PokemonEntity = genetic.genome_id_to_pokemon[genome.selected_parent[0]]
                    pok = PokemonEntity(parent.index, genome, config)
                    pok.entity = parent.entity.copy()
                    pok.entity[0] = parent.entity[0]
                    pok.entity[0].moves = parent.entity[0].moves.copy()
                    pok.entity[0].nature = parent.entity[0].nature
                    pok.entity[0].ability = parent.entity[0].ability
                    newborn_n = newborn_n +1
                    if get_U() < 0.69 and genome.selected_parent[1] in genetic.genome_id_to_pokemon:
                        parent2: PokemonEntity = genetic.genome_id_to_pokemon[genome.selected_parent[1]]
                        genetic.crossover(pok, parent2)
                    pok.lvl = 1
                    pok.entity[0].level = 1
                    pok.xp = 0
                    genetic.genome_id_to_pokemon[genome_id] = pok
                    pokemons.append(pok)
                        
                else:
                    pok = PokemonEntity(genetic.index_generator(), genome, config)
                    genetic.genome_id_to_pokemon[genome_id] = pok
                    pokemons.append(pok)
                    created_n = created_n +1
            
            U = get_U()
            p = pok.temperature()
            if U < p and pok.genome.fitness > 0.005:
                mut_n = mut_n + genetic.mutate(pok)
           
        print("Species:",pokemons[0].entity[0].species)
        print("Seleccionados:", selected_n,"pokemons de", len(pokemons))
        print("Nacidos:", newborn_n,"pokemons de", len(pokemons))
        print("Creados:", created_n,"pokemons de", len(pokemons))
        print("Mutados:", mut_n,"pokemons de", len(pokemons))
        
        max_fitness = 0.0
        # genetic.pkmn_logic.pokemon_bipartite_reg_matches(pokemons, 30)
        # print("iteration started")
        for i,pkmn in enumerate(pokemons):
            pkmn: PokemonEntity
            pkmn.used_moves = {}
            pkmn.age = pkmn.age +1
            
            if (np.mod(i,10)==0):
                print("pokemon", i)
            pkmn:PokemonEntity
            pkmn.won_battles = 0
            pkmn.total_battles = 0
            for _ in range(70):
                foe: PokemonEntity = np.random.choice(genetic.available_pokemons)
                foe.lvl = pkmn.lvl
                foe.entity[0].level = pkmn.lvl
                pkmn.won_battles = pkmn.won_battles +genetic.pkmn_logic.battle(pkmn, foe)
                pkmn.total_battles = pkmn.total_battles +1
            
            pkmn.update_fitness(genetic.age)
            if pkmn.genome.fitness > max_fitness: #(pkmn.lvl > 90) or (pkmn.won_battles > 46):
                max_fitness = pkmn.genome.fitness
                if max_fitness > genetic.m_fitness:
                    genetic.m_fitness = max_fitness
                    print()
                    print("pkmn index",i)
                    print("pkmn lvl", pkmn.lvl)
                    print("pkmn age", pkmn.age)
                    print("pkmn used moves", str(pkmn.used_moves))
                    print("battles won",pkmn.won_battles,"/",pkmn.total_battles)
                    print("pkmn fit", pkmn.genome.fitness)
                if genetic.ax_method:
                    genetic.ax_method(max_fitness)
            
    def run(gens):
        pop = neat.population.Population(CONFIG)
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.reporting.StdOutReporter(True))

        winner = pop.run(eval, gens)
        return winner, stats
    
    training = [
    3,    # Venusaur
    6,    # Charizard
    9,    # Blastoise
    65,   # Alakazam
    68,   # Machamp
    94,   # Gengar
    130,  # Gyarados
    143,  # Snorlax
    149,  # Dragonite
    150,  # Mewtwo
    248,  # Tyranitar
    257,  # Blaziken
    282,  # Gardevoir
    306,  # Aggron
    310,  # Manectric
    373,  # Salamence
    376,  # Metagross
    384,  # Rayquaza
    445,  # Garchomp
    448,  # Lucario
    462,  # Magnezone
    475,  # Gallade
    530,  # Excadrill
    609,  # Chandelure
    635,  # Hydreigon
    681,  # Aegislash
    701,  # Hawlucha
    724,  # Decidueye
    784,  # Kommo-o
    798   # Kartana
]

    
    for pkmn_index in training:#range(1, 152): # GEN 1 starters of the games
        genetic.age = 0
        genetic.genome_id_to_pokemon = {}
        genetic.index_generator = lambda:pkmn_index
        genetic.m_fitness = 0.0
        # add the already saved pokemons to the posible foes
        for pat in Path("./pokemons/to_train/").iterdir():
            genetic.available_pokemons.append(load_pokemon(pat))
        # pkmn_index = np.random.randint(808)
        print("\ntraining",PokemonEntity(pkmn_index,None, None, False).entity[0].species, "with:")
        for pokemn in genetic.available_pokemons:
            print(" -",pokemn.entity[0].species)
        def generator():
            return pkmn_index
        genetic.genome_id_to_pokemon = {}
        genetic.index_generator = generator
        
        top_genome = run(NO_OF_GENS)[0]
        winner: PokemonEntity = genetic.genome_id_to_pokemon[top_genome.key]
        
        print()
        print("Results:")
        print("Specie:", winner.entity[0].species)
        print("Won:",winner.won_battles, "from", winner.total_battles)
        print("Level:",winner.lvl)
        print("Age:",winner.age)
        print("Used Moves:",str(winner.used_moves))
        print("Fitness:", winner.genome.fitness)
        print("Ability:", winner.entity[0].ability)
        print("Nature:", winner.entity[0].nature)
        print("Moves:", *winner.entity[0].moves, sep = "\n - ")
        print()
        # print("press a key")
        # input()
        save_pokemon(winner)
        print("pokemon_saved")