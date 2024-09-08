from data import dex
import sim.sim as sim
import sim.player as player
import pandas as pd
from tools.pick_six import generate_team_pok
import numpy as np
import random
from sim.structs import Pokemon

import pickle
import neat
import neat.nn
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

COOLER = 11
TURN_BONUS = 30

VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"          

INPUT_COORDINATES = [(float(j-6), float(-2)) for j in range(13)]
OUTPUT_COORDINATES = [(float(x-3), 2.0) for x in list(range(3))+list(range(4,7))]

SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

NO_OF_GENS = 9999
# MATCHES_BY_GEN = 1
# MATCHES_TO_RECOMBINE = 10

def get_U():
    return np.random.uniform()

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 3.0,
            "activation": "sigmoid"}

DYNAMIC_PARAMS = params(VERSION)

CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn')

class PokemonEco:
    def __init__(self):
        self.index_to_id_pokemon = dex.index_to_id_pokemon
        self.id_to_index_pokemon = {v: k for k, v in dex.index_to_id_pokemon.items()}


class PokemonEntity:
    def __init__(self, index, genome, config, has_ai = True, brain_size = "S", level = 1):
        self.entity, self.other_moves, self.other_natures, self.other_abilities = generate_team_pok(index)
        self.entity = sim.dict_to_team_set(self.entity)
        self.lvl = level
        self.entity[0].level = self.lvl
        self.xp = 0
        self.brAInNEAT = None
        self.brain_size_text = None
        self.genome = genome
        self.config = config
        self.network = None
        if has_ai:
            self.set_ai()
            self.genome.fitness = 0.001
            self.brain_size_text = "small" if brain_size == "S" else "medium" if brain_size == "M" else "large"
        
    def set_ai(self):
        cppn = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.network = ESNetwork(SUBSTRATE, cppn, DYNAMIC_PARAMS)
        net = self.network.create_phenotype_network()
        self.brAInNEAT = net
        net.reset()
        
    def choose_random(self):
        return np.random.randint(-1,5)
    
    def choose(self, myself: Pokemon, rival: Pokemon):
        """
        inputs would be:
        minepkmn:
            hp / maxhp (0,1)
            lvl / 100 (0,1)
            type1
            type2
        rivalpkmn
            hp / maxhp
            lvl / 100
            type1
            type2
        bias
        
        outputs
        pass -1
        attacks
            0, 1, 2, 3
        run 4
        """
        A_HP = (myself.hp / myself.maxhp ,)
        A_LV = (myself.level / 100 ,)
        A_type1 = tuple([float(np.sin(hash(myself.types[0]))), float(np.cos(hash(myself.types[0])))])
        A_type2 = (0.0, 0.0) if len(myself.types)<=1 else tuple([float(np.sin(hash(myself.types[1]))), float(np.cos(hash(myself.types[1])))])
        bias = (1.0,)
        B_HP = (rival.hp / rival.maxhp ,)
        B_LV = (rival.level / 100 ,)
        B_type1 = tuple([float(np.sin(hash(rival.types[0]))), float(np.cos(hash(rival.types[0])))])
        B_type2 = (0.0, 0.0) if len(rival.types)<=1 else tuple([float(np.sin(hash(rival.types[1]))), float(np.cos(hash(rival.types[1])))])
        
        net_input = A_HP + A_LV + A_type1 + A_type2 + bias + B_HP + B_LV + B_type1 + B_type2
        
        # print(net_input)
        
        self.brAInNEAT.reset()
        net_output = None
        for _ in range(self.network.activations):
            net_output = self.brAInNEAT.activate(net_input)
        # print("A")
        return net_output.index(max(net_output)) -1
        # exit()
        # return self.choose_random()
        
    
    def fitness(self):
        return (self.lvl/100)**(5/7)
    def fitness_d(self):
        return self.lvl/100
    def temperature(self):
        return (1 - self.genome.fitness)**COOLER
    def update_fitness(self):
        self.genome.fitness = self.fitness_d()

def save_pokemon(pokemon: PokemonEntity):
    with open(f'pokemons/{pokemon.entity[0].species}_{pokemon.entity[0].nature}_{VERSION_TEXT}_{int(pokemon.genome.fitness*100)}.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
    filename
    

class ExperienceHandler:
    def __init__(self):
        
        self.pokemon_index_to_xp_data = {} # dict of index to dicts holding xp data by pokemon

        pokemon_exp_yield = (
            pd.read_csv("data/pokemon_ev_yields_ok.csv")[["Number", "Exp. Yield"]]
        )[:849]
        for row in pokemon_exp_yield.iterrows():
            if not int(row[1]["Number"]) in self.pokemon_index_to_xp_data:
                self.pokemon_index_to_xp_data[int(row[1]["Number"])] = {}
                (self.pokemon_index_to_xp_data[int(row[1]["Number"])])["xp_base"] = float(
                    row[1]["Exp. Yield"]
                )

        pokemon_exp_type = (
            pd.read_csv("data/pokemon_experience_types.csv")[["Number", "Experience Type"]]
        )[:807]

        for row in pokemon_exp_type.iterrows():
            if int(row[1]["Number"]) in self.pokemon_index_to_xp_data:
                (self.pokemon_index_to_xp_data[int(row[1]["Number"])])["xp_type"] = str(
                    row[1]["Experience Type"]
                )

        def toin(x):
            xn = str.replace(x,',','')
            return int(xn)

        pokemon_exp_to_level_up = (
            pd.read_csv("data/experience_table.csv")[['Erratic', 'Fast', 'Medium Fast', 'Medium Slow', 'Slow',
            'Fluctuating']]
        )
        self.pokemon_exp_to_level_up_by_type = {} # this holds the leveling requirements for each experience type
        for column in pokemon_exp_to_level_up.columns:
            self.pokemon_exp_to_level_up_by_type[column] = [x for x in map(toin, list(pokemon_exp_to_level_up[column]))]

    def experience_gain(self, winner_lvl, loser_lvl, loser_index):
        b = self.pokemon_index_to_xp_data[loser_index]["xp_base"]
        l = (b * loser_lvl)/5
        r = ((2*loser_lvl+10)/(loser_lvl + winner_lvl + 10))**2.5
        x = l * r + 1
        return int(x)
    
    def leveling_handler(self, pkmn_id, pkmn_lvl, pkmn_total_xp):
        pkmn_ind = PokemonEco().id_to_index_pokemon[pkmn_id]
        cum_lvl = self.pokemon_exp_to_level_up_by_type[self.pokemon_index_to_xp_data[pkmn_ind]["xp_type"]]
        for lvl in range(pkmn_lvl,100):
            if pkmn_total_xp >= cum_lvl[lvl]:
                pkmn_lvl = pkmn_lvl + 1
                # print("leveled up to level:",pkmn_lvl)
            else:
                break
        return pkmn_lvl

class BasicPkmnLogic:
    def __init__(self) -> None:
        self.experience_handler = ExperienceHandler()
        self.ecosystem = PokemonEco()
        
    def try_run(self, pkmn1: Pokemon, pkmn2: Pokemon):
        U = get_U()
        r = pkmn1.stats.speed/pkmn2.stats.speed
        p = (np.tanh(3*(r - 0.95))+1)/2
        return U < p
            
        
    def battle(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity, debug = False):
        exp = True
        battle = sim.Battle('single', 'A', pkmn1.entity, 'B', pkmn2.entity, debug=debug)
        # print(battle.p1.active_pokemon[0].types[0])
        # print(battle.p1.active_pokemon[0].types[1])
        runaway = 1
        MAX_TURNS = 500
        while not battle.ended:
            if battle.turn > 500:
                exp = False
                break
            pkmn1_choice = pkmn1.choose(battle.p1.active_pokemon[0], battle.p2.active_pokemon[0])
            pkmn2_choice = pkmn2.choose(battle.p2.active_pokemon[0], battle.p1.active_pokemon[0])
            if not (pkmn1_choice < 4) and not (pkmn2_choice < 4): #both run
                exp = False
                break
            elif not (pkmn1_choice < 4): #pkmn1 run
                success_run = self.try_run(battle.p1.active_pokemon[0],battle.p2.active_pokemon[0])
                if success_run:
                    battle.winner = 'p2'
                    exp = False
                    break
                else:
                    battle.p1.choice = player.Decision('pass', 0)
                    battle.p2.choice = player.Decision('move', pkmn2_choice)
            elif not (pkmn2_choice < 4): #pkmn2 run
                success_run = self.try_run(battle.p2.active_pokemon[0],battle.p1.active_pokemon[0])
                if success_run:
                    battle.winner = 'p1'
                    exp = False
                    break
                else:
                    battle.p1.choice = player.Decision('move', pkmn1_choice)
                    battle.p2.choice = player.Decision('pass', 0)            
            else:
                battle.p1.choice = player.Decision('move', pkmn1_choice)
                battle.p2.choice = player.Decision('move', pkmn2_choice)
            
            if pkmn1_choice == -1:
                battle.p1.choice = player.Decision('pass', 0)
            if pkmn2_choice == -1:
                battle.p2.choice = player.Decision('pass', 0)
            
            # Decision('pass', 0)
            # print("B")
            sim.do_turn(battle)
            
        if exp:
            if battle.winner == 'p1':
                winner = pkmn1
                loser = pkmn2
            if battle.winner == 'p2':
                winner = pkmn2
                loser = pkmn1
            # print(battle.log)
            # print(battle.winner)
            loser_index = self.ecosystem.id_to_index_pokemon[loser.entity[0].species]
            # print(battle.turn)
            modifier = int(max(TURN_BONUS/battle.turn, 1))
            
            winner.xp = winner.xp + modifier*self.experience_handler.experience_gain(winner.lvl, loser.lvl, loser_index)
            winner.lvl = self.experience_handler.leveling_handler(winner.entity[0].species, winner.lvl, winner.xp)
            winner.entity[0].level = winner.lvl
        
    def pokemon_match_old(self, pokemon_list):
        # print(len(pokemon_list))
        # Shuffle the list to randomize the order
        random.shuffle(pokemon_list)
        for i in range(0, len(pokemon_list), 2):
            # print("battle no:", i)
            if i + 1 < len(pokemon_list):  # Ensure there is a pair
                pokemon1 = pokemon_list[i]
                pokemon2 = pokemon_list[i + 1]
                self.battle(pokemon1, pokemon2)
                
    def pokemon_match(self, pokemon_list):
        for i in range(len(pokemon_list)):
            for j in range(i+1, len(pokemon_list)):
                pokemon1 = pokemon_list[i]
                pokemon2 = pokemon_list[j]
                self.battle(pokemon1, pokemon2)
    
    def pokemon_bipartite_reg_matches(self, pokemon_list, t):
        random.shuffle(pokemon_list)
        A = pokemon_list[:int(len(pokemon_list)/2)]
        B = pokemon_list[int(len(pokemon_list)/2):]
        for i in range(int(len(pokemon_list)/2)):
            for j in range(t):
                self.battle(A[i], B[np.mod(j, int(len(pokemon_list)/2))])
                
class GeneticEvolution:
    def __init__(self, index_generator):
        self.index_generator = index_generator
        self.genome_id_to_pokemon = {}
        self.pkmn_logic = BasicPkmnLogic()
        
    def mutate(self, pkmn: PokemonEntity):
        no_of_muts = 1
        U1 = get_U()
        if U1 < 0.05:
            no_of_muts = 5
        elif U1 < 0.13:
            no_of_muts = 4
        elif U1 < 0.23:
            no_of_muts = 3 
        elif U1 < 0.47:
            no_of_muts = 2
        
        for _ in range(no_of_muts):
            U2 = get_U()
            if U2 < 0.6: # mutate move
                index_out = np.random.randint(len(pkmn.entity[0].moves))
                move_out = pkmn.entity[0].moves[index_out]
                index_in = np.random.randint(len(pkmn.other_moves))
                move_in = pkmn.other_moves[index_in]
                pkmn.entity[0].moves[index_out] = move_in
                pkmn.other_moves[index_in] = move_out
                
            elif U2<0.9: #mutate ability
                ability_out = pkmn.entity[0].ability
                ability_in_index = np.random.randint(len(pkmn.other_abilities))
                ability_in = pkmn.other_abilities[ability_in_index]
                pkmn.other_abilities[ability_in_index]= ability_out
                pkmn.entity[0].ability = ability_in
                
            else: #mutate nature
                pass
                # nature_out = pkmn.entity[0].nature
                # nature_in_index = np.random.randint(len(pkmn.other_abilities))
                # nature_in = pkmn.other_abilities[nature_in_index]
                # pkmn.other_abilities[nature_in_index]= nature_out
                # pkmn.entity[0].nature = nature_in
        
    def reproduce(pkmn_x: PokemonEntity, pkmn_y: PokemonEntity):
        "crossover"
        pass
    
    # def select(list, fitnes_fnc): MADE BY LIB
    #     pass

    

    
    def eval_fitness_matches(self, genomes, config):
        pokemons = []
        mut_n = 0
        for genome_id, genome in genomes:
            pok = None
            if genome_id in self.genome_id_to_pokemon:
                pok : PokemonEntity = self.genome_id_to_pokemon[genome_id]
                pok.lvl = 1
                pok.entity[0].level = 1
                pok.xp = 0
                pok.genome = genome
                pok.set_ai()
                pokemons.append(pok)
            else:
                pok = PokemonEntity(self.index_generator(), genome, config)
                self.genome_id_to_pokemon[genome_id] = pok
                pokemons.append(pok)
            U = get_U()
            p = pok.temperature()
            if U < p:
                # print(U, "<", p)
                mut_n = mut_n + 1
                self.mutate(pok)
            # print(genome_id)
        
        print("Mutando:", mut_n,"pokemons de", len(pokemons))
        
        self.pkmn_logic.pokemon_bipartite_reg_matches(pokemons, 40)
        # self.pkmn_logic.pokemon_match(pokemons)
        
        # for match_no in range(MATCHES_BY_GEN):
        #     print("match no:", match_no)
        #     self.pkmn_logic.pokemon_match(pokemons)
        #     if np.mod(match_no, MATCHES_TO_RECOMBINE) == 0:
        #         pass
            
        for pokemon in pokemons:
            pokemon:PokemonEntity
            pokemon.update_fitness()
            # print(pokemon.genome.fitness)
            # print(pokemon.lvl)
        
    def run(self, gens, version):
        pop = neat.population.Population(CONFIG)
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.reporting.StdOutReporter(True))

        global DYNAMIC_PARAMS
        DYNAMIC_PARAMS = params(version)

        winner = pop.run(self.eval_fitness_matches, gens)
        # print(f"es_hyperneat_{VERSION_TEXT} done")
        return winner, stats
    
    
if __name__ == "__main__":
    def gen():
        U = get_U()
        if U < 0.33333333:
            return 3
        elif U < 0.66666666:
            return 6
        else:
            return 9
                
    genetic = GeneticEvolution(gen)
    top_genome = genetic.run(NO_OF_GENS, VERSION)[0]
    winner: PokemonEntity = genetic.genome_id_to_pokemon[top_genome.key]
    print("Fitness:", winner.genome.fitness)
    print("Specie:", winner.entity[0].species)
    print("Ability:", winner.entity[0].ability)
    print("Nature:", winner.entity[0].nature)
    print("Moves:", *winner.entity[0].moves, sep = "\n - ")
    
    
    
    # a = PokemonEntity(1,None,None, level= 80, has_ai=False)
    # print(a.fitness())
    # exit()
    
    # pok1 = PokemonEntity(151, None,None,False, level=50)
    # pok2 = PokemonEntity(3, None,None,False, level=50)
    # print(pok1.entity[0].level)
    # BasicPkmnLogic().battle(pok1, pok2)#, True)
    
    # pok = PokemonEntity(151, None,None,False, level=50)
    # print(pok.entity[0])

    # print(bulb.other_moves)
    # print(bulb.other_natures)
    # print(bulb.other_abilities)

    # xp_h = ExperienceHandler()
    # print(xp_h.experience_gain(100, 100, 807))
    # print(xp_h.leveling_handler(807,100,xp_h.experience_gain(100, 100, 84)))
    # xp_h.leveling_handler(12,1, 26)
    # for i in pokemon_index_to_xp_data:
    #     print(i, "->", pokemon_index_to_xp_data[i])

    # print(pokemon_exp_to_level_up)

    # for typ in pokemon_exp_to_level_up_by_type:
    #     print(typ, "->", pokemon_exp_to_level_up_by_type[typ])
    #     print()

    # a = True
    # for x in pokemon_index_to_xp_data:
    #     if not pokemon_index_to_xp_data[x]["xp_type"] in pokemon_exp_to_level_up_by_type:
    #         a = False
    # print(a)

    # print(pokemon_exp_type)
    # print(pokedex[all_pokemons_names[37]])
    # print(len(all_pokemons_names))
    # print(all_pokemons_names[-1])
    # print(id_to_index_pokemon)
    # print(index_to_id_pokemon)
