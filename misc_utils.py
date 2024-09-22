from data import dex
import sim.sim as sim
import sim.player as player
import pandas as pd
from tools.pick_six import generate_team_pok
import numpy as np
import random
from sim.structs import Pokemon

from pathlib import Path
import pickle
import neat
import neat.nn
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

COOLER = 3
TURN_BONUS = 10

BRAIN_SIZE = "S"
BRAIN_SIZE_TEXT = "small" if BRAIN_SIZE == "S" else "medium" if BRAIN_SIZE == "M" else "large"          

INPUT_COORDINATES = [(float(j-6), float(-2)) for j in range(13)]
OUTPUT_COORDINATES = [(float(x-3), 2.0) for x in list(range(3))+list(range(4,7))]

SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

NO_OF_GENS = None

# MATCHES_BY_GEN = 1
# MATCHES_TO_RECOMBINE = 10

def get_U():
    return np.random.uniform()

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def sample_from_distribution(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 2 if version == "S" else 3 if version == "M" else 5,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 3.0,
            "activation": "sigmoid"}

DYNAMIC_PARAMS = params(BRAIN_SIZE)

CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn')

class PokemonEco:
    def __init__(self):
        self.index_to_id_pokemon = dex.index_to_id_pokemon
        self.id_to_index_pokemon = {v: k for k, v in dex.index_to_id_pokemon.items()}


class PokemonEntity:
    def __init__(self, index, genome, config, has_ai = True, level = 1):
        self.index = index
        self.entity, self.other_moves, self.other_natures, self.other_abilities = generate_team_pok(index)
        self.entity = sim.dict_to_team_set(self.entity)
        self.lvl = level
        self.entity[0].level = self.lvl
        self.xp = 0
        self.brAInNEAT = None
        self.genome = genome
        self.config = config
        self.network = None
        self.has_ai = has_ai
        if has_ai:
            self.set_ai()
            self.genome.fitness = 0.001
            
        self.won_battles = 0
        self.total_battles = 0
        self.used_moves = set()
        self.age = 0
        
    def set_ai(self):
        cppn = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.network = ESNetwork(SUBSTRATE, cppn, DYNAMIC_PARAMS)
        net = self.network.create_phenotype_network()
        self.brAInNEAT = net
        net.reset()
        
    def choose_random(self):
        return np.random.randint(-1,5)
    
    def ask_for_choice(self, myself: Pokemon, rival: Pokemon):
        if self.genome:
            return self.choose(myself, rival)
        else:
            print("Select an option:")
            print("> -1 pass")
            j = 0
            for i in range(len(self.entity[0].moves)):
                print(">",i, self.entity[0].moves[i])
                j = i+1
            print(">", j, "run")
            chs = input()
            return int(chs)
                
    
    def choose(self, myself: Pokemon, rival: Pokemon):
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
                
        self.brAInNEAT.reset()
        net_output = None
        for _ in range(self.network.activations):
            net_output = self.brAInNEAT.activate(net_input)
        return net_output.index(max(net_output)) -1    
        # return sample_from_distribution(softmax(net_output)) -1 
    
    def fitness(self, age):
        x = self.lvl/100
        if self.total_battles > 0:
            x = 0.4*x + 0.6*(self.won_battles/self.total_battles)
        p = (3/(age + 2.3))**2
        return p*((len(self.used_moves)/4)**2.9) + (1-p)*x
    
    def fitness_d(self):
        return self.lvl/100
    def temperature(self):
        return (1 - self.genome.fitness)**COOLER
    def update_fitness(self, age):
        self.genome.fitness = self.fitness(age)

def save_pokemon(pokemon: PokemonEntity):
    with open(f'pokemons/{pokemon.entity[0].species}_{pokemon.entity[0].nature}_{BRAIN_SIZE_TEXT}_{int(pokemon.genome.fitness*100)}.pkl', 'wb') as output:
        pickle.dump(pokemon, output, pickle.HIGHEST_PROTOCOL)
    with open(f'pokemons/pokemon_data/{pokemon.entity[0].species}_{pokemon.entity[0].nature}_{BRAIN_SIZE_TEXT}_{int(pokemon.genome.fitness*100)}.txt', 'w') as file:
        file.write(f"Specie:{pokemon.entity[0].species}\n")
        file.write(f"Won:{pokemon.won_battles}/{pokemon.total_battles} battles\n")
        file.write(f"Level:{pokemon.lvl}\n")
        file.write(f"Fitness:{pokemon.genome.fitness}\n")
        file.write(f"Ability:{pokemon.entity[0].ability}\n")
        file.write(f"Nature:{pokemon.entity[0].nature}\n")
        file.write(f"Moves:{str(pokemon.entity[0].moves)}\n")
        file.write(f"Most Used Moves:{str(pokemon.used_moves)}\n")
    
    
def load_pokemon(path):
    pokemon: PokemonEntity = None
    with open(path, 'rb') as input_data:
        pokemon = pickle.load(input_data)
    return pokemon


    

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
    
    def base_stats(self, pkmn: Pokemon):
        a = pkmn.stats.hp
        b = pkmn.stats.attack
        c = pkmn.stats.defense
        d = pkmn.stats.specialattack
        e = pkmn.stats.specialdefense
        f = pkmn.stats.speed
        return (a + b + c + d + e + f)/6
    
    def stats_mod(self, pkmn1: Pokemon, pkmn2: Pokemon):
        a = self.base_stats(pkmn1)
        b = self.base_stats(pkmn2)
        x = b/a
        return (((np.tanh(13*(x-1.1))+1)/2)+0.7*x)*1.5**x
    
class BasicPkmnLogic:
    def __init__(self) -> None:
        self.experience_handler = ExperienceHandler()
        self.ecosystem = PokemonEco()
        self.lb_log = None
        
        self.manual_battle = None
        self.manual_battle_exp = False
        
    def try_run(self, pkmn1: Pokemon, pkmn2: Pokemon):
        U = get_U()
        r = pkmn1.stats.speed/pkmn2.stats.speed
        p = (np.tanh(3*(r - 0.95))+1)/2
        return U < p
            
    def manual_battle_create(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity):
        self.manual_battle_exp = True
        self.manual_battle = sim.Battle('single', 'Piad', pkmn1.entity, 'Daniel', pkmn2.entity)
        
    def manual_battle_do_turn(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity, my_choice: int):
        # print(my_choice)
        pkmn1_choice = my_choice
        pkmn2_choice = pkmn2.ask_for_choice(self.manual_battle.p2.active_pokemon[0], self.manual_battle.p1.active_pokemon[0])
        if not (pkmn1_choice < 4) and not (pkmn2_choice < 4): #both run
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" runs")
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" runs")
            self.manual_battle.log.append("both pokemons runs so the self.manual_battle ends")
            self.manual_battle_exp = False
            return True
        elif not (pkmn1_choice < 4): #pkmn1 run
            success_run = self.try_run(self.manual_battle.p1.active_pokemon[0],self.manual_battle.p2.active_pokemon[0])
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" prepares to run")
            if success_run:
                self.manual_battle.log.append("succesfully")
                self.manual_battle.winner = 'p2'
                self.manual_battle_exp = False
                return True
            else:
                self.manual_battle.log.append("but failed")
                self.manual_battle.p1.choice = player.Decision('pass', 0)
                self.manual_battle.p2.choice = player.Decision('move', pkmn2_choice)
        elif not (pkmn2_choice < 4): #pkmn2 run
            success_run = self.try_run(self.manual_battle.p2.active_pokemon[0],self.manual_battle.p1.active_pokemon[0])
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" prepares to run")
            if success_run:
                self.manual_battle.log.append("succesfully")
                self.manual_battle.winner = 'p1'
                self.manual_battle_exp = False
                return True
            else:
                self.manual_battle.log.append("but failed")
                self.manual_battle.p1.choice = player.Decision('move', pkmn1_choice)
                self.manual_battle.p2.choice = player.Decision('pass', 0)            
        else:
            self.manual_battle.p1.choice = player.Decision('move', pkmn1_choice)
            self.manual_battle.p2.choice = player.Decision('move', pkmn2_choice)
        
        if pkmn1_choice == -1:
            self.manual_battle.p1.choice = player.Decision('pass', 0)
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" passes his turn")
        if pkmn2_choice == -1:
            self.manual_battle.p2.choice = player.Decision('pass', 0)
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" passes his turn")

        sim.do_turn(self.manual_battle)
        
        if self.manual_battle.p1.active_pokemon[0].hp <=0:
            self.manual_battle.log.append(self.manual_battle.p1.active_pokemon[0].species+" has fainted")
            return True
        elif self.manual_battle.p2.active_pokemon[0].hp <=0:
            self.manual_battle.log.append(self.manual_battle.p2.active_pokemon[0].species+" has fainted")
            return True
        
        self.manual_battle.log.append(f"status: {pkmn1.entity[0].species} hp:{self.manual_battle.p1.active_pokemon[0].hp}/{self.manual_battle.p1.active_pokemon[0].maxhp} and {pkmn2.entity[0].species} hp:{self.manual_battle.p2.active_pokemon[0].hp}/{self.manual_battle.p2.active_pokemon[0].maxhp}")
        if self.manual_battle.ended:
        # print()
        # print(self.manual_battle.log)
        # input()
        # print()
            self.lb_log = self.manual_battle.log
            
            if self.manual_battle_exp:
                if self.manual_battle.winner == 'p1':
                    winner = pkmn1
                    loser = pkmn2
                    # a = self.experience_handler.base_stats(self.manual_battle.p1.active_pokemon[0])
                    # b = self.experience_handler.base_stats(self.manual_battle.p2.active_pokemon[0])
                    # stat_modifier = self.experience_handler.stats_mod(self.manual_battle.p1.active_pokemon[0],self.manual_battle.p2.active_pokemon[0])
                if self.manual_battle.winner == 'p2':
                    winner = pkmn2
                    loser = pkmn1
                    # a = self.experience_handler.base_stats(self.manual_battle.p2.active_pokemon[0])
                    # b = self.experience_handler.base_stats(self.manual_battle.p1.active_pokemon[0])
                    # stat_modifier = self.experience_handler.stats_mod(self.manual_battle.p2.active_pokemon[0],self.manual_battle.p1.active_pokemon[0])
                # print()
                # print(a)
                # print(b)
                # print(stat_modifier)
                # print()
                loser_index = self.ecosystem.id_to_index_pokemon[loser.entity[0].species]
                modifier = (self.manual_battle.turn+TURN_BONUS)/self.manual_battle.turn
                
                winner.xp = winner.xp + int(modifier*self.experience_handler.experience_gain(winner.lvl, loser.lvl, loser_index))
                winner.lvl = self.experience_handler.leveling_handler(winner.entity[0].species, winner.lvl, winner.xp)
                winner.entity[0].level = winner.lvl
                return True
            
    def battle(self, pkmn1: PokemonEntity, pkmn2: PokemonEntity, debug = False):
        exp = True
        battle = sim.Battle('single', 'A', pkmn1.entity, 'B', pkmn2.entity, debug=debug)  
        MAX_TURNS = 500
        
        while not battle.ended:
            if battle.turn > 500:
                exp = False
                break
            battle.log.append(f"status: {pkmn1.entity[0].species} hp:{battle.p1.active_pokemon[0].hp}/{battle.p1.active_pokemon[0].maxhp} and {pkmn2.entity[0].species} hp:{battle.p2.active_pokemon[0].hp}/{battle.p2.active_pokemon[0].maxhp}")
            pkmn1_choice = pkmn1.ask_for_choice(battle.p1.active_pokemon[0], battle.p2.active_pokemon[0])
            pkmn2_choice = pkmn2.ask_for_choice(battle.p2.active_pokemon[0], battle.p1.active_pokemon[0])
            if not (pkmn1_choice < 4) and not (pkmn2_choice < 4): #both run
                battle.log.append(battle.p1.active_pokemon[0].species+" runs")
                battle.log.append(battle.p2.active_pokemon[0].species+" runs")
                battle.log.append("both pokemons runs so the battle ends")
                exp = False
                break
            elif not (pkmn1_choice < 4): #pkmn1 run
                success_run = self.try_run(battle.p1.active_pokemon[0],battle.p2.active_pokemon[0])
                battle.log.append(battle.p1.active_pokemon[0].species+" prepares to run")
                if success_run:
                    battle.log.append("succesfully")
                    battle.winner = 'p2'
                    exp = False
                    break
                else:
                    battle.log.append("but failed")
                    battle.p1.choice = player.Decision('pass', 0)
                    battle.p2.choice = player.Decision('move', pkmn2_choice)
            elif not (pkmn2_choice < 4): #pkmn2 run
                success_run = self.try_run(battle.p2.active_pokemon[0],battle.p1.active_pokemon[0])
                battle.log.append(battle.p2.active_pokemon[0].species+" prepares to run")
                if success_run:
                    battle.log.append("succesfully")
                    battle.winner = 'p1'
                    exp = False
                    break
                else:
                    battle.log.append("but failed")
                    battle.p1.choice = player.Decision('move', pkmn1_choice)
                    battle.p2.choice = player.Decision('pass', 0)            
            else:
                battle.p1.choice = player.Decision('move', pkmn1_choice)
                battle.p2.choice = player.Decision('move', pkmn2_choice)
            
            if pkmn1_choice == -1:
                battle.p1.choice = player.Decision('pass', 0)
                battle.log.append(battle.p1.active_pokemon[0].species+" passes his turn")
            if pkmn2_choice == -1:
                battle.p2.choice = player.Decision('pass', 0)
                battle.log.append(battle.p2.active_pokemon[0].species+" passes his turn")

            if pkmn1_choice in range(4):
                pkmn1.used_moves.add(pkmn1.entity[0].moves[pkmn1_choice])
            # if pkmn2_choice in range(4):
            #     pkmn2.used_moves.add(pkmn2.entity[0].moves[pkmn2_choice])
                
            sim.do_turn(battle)
        if battle.p1.active_pokemon[0].hp <=0:
            battle.log.append(battle.p1.active_pokemon[0].species+" has fainted")
        if battle.p2.active_pokemon[0].hp <=0:
            battle.log.append(battle.p2.active_pokemon[0].species+" has fainted")
        
        # print()
        # print(battle.log)
        # input()
        # print()
        self.lb_log = battle.log
        
        if exp:
            if battle.winner == 'p1':
                winner = pkmn1
                loser = pkmn2
                # a = self.experience_handler.base_stats(battle.p1.active_pokemon[0])
                # b = self.experience_handler.base_stats(battle.p2.active_pokemon[0])
                # stat_modifier = self.experience_handler.stats_mod(battle.p1.active_pokemon[0],battle.p2.active_pokemon[0])
            if battle.winner == 'p2':
                winner = pkmn2
                loser = pkmn1
                # a = self.experience_handler.base_stats(battle.p2.active_pokemon[0])
                # b = self.experience_handler.base_stats(battle.p1.active_pokemon[0])
                # stat_modifier = self.experience_handler.stats_mod(battle.p2.active_pokemon[0],battle.p1.active_pokemon[0])
            # print()
            # print(a)
            # print(b)
            # print(stat_modifier)
            # print()
            loser_index = self.ecosystem.id_to_index_pokemon[loser.entity[0].species]
            modifier = (battle.turn+TURN_BONUS)/battle.turn
            
            winner.xp = winner.xp + int(modifier*self.experience_handler.experience_gain(winner.lvl, loser.lvl, loser_index))
            winner.lvl = self.experience_handler.leveling_handler(winner.entity[0].species, winner.lvl, winner.xp)
            winner.entity[0].level = winner.lvl
        if battle.winner == 'p1':
            return 1
        else:
            return 0
        
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
    def __init__(self, index_generator, reproduce = False):
        self.index_generator = index_generator
        self.genome_id_to_pokemon = {}
        self.pkmn_logic = BasicPkmnLogic()
        self.available_pokemons = []
        self.reproduce = reproduce
        self.ax_method = None
        self.m_fitness = 0.0
        self.age = 0
        
    def mutate(self, pkmn: PokemonEntity):
        # no_of_muts = 1
        # U1 = get_U()
        # if U1 < 0.05:
        #     no_of_muts = 5
        # elif U1 < 0.13:
        #     no_of_muts = 4
        # elif U1 < 0.23:
        #     no_of_muts = 3 
        # elif U1 < 0.47:
        #     no_of_muts = 2
        mutates = 0
        tc = 0
        while get_U() < pkmn.temperature():
            tc = tc +1
            if tc > 7:
                break
            # print("mutate",tc)
            mutates = 1
            U2 = get_U()
            if U2 < 0.6 and len(pkmn.entity[0].moves)>0 and len(pkmn.other_moves)>0: # mutate move
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
                # self.mutate(pkmn)
                
                # nature_out = pkmn.entity[0].nature
                # nature_in_index = np.random.randint(len(pkmn.other_abilities))
                # nature_in = pkmn.other_abilities[nature_in_index]
                # pkmn.other_abilities[nature_in_index]= nature_out
                # pkmn.entity[0].nature = nature_in
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
                # print("mixed moves properly")
                break
            else:
                # print("not mixed properly")
                if get_U()<0.1:
                    # print("outa loop")
                    break
    
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
        print()
        
        self.pkmn_logic.pokemon_bipartite_reg_matches(pokemons, 40)
            
        for pokemon in pokemons:
            pokemon:PokemonEntity
            pokemon.update_fitness(self.age)
            
        
    def run_matches_itself(self, gens, version):
        pop = neat.population.Population(CONFIG)
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.reporting.StdOutReporter(True))

        global DYNAMIC_PARAMS
        DYNAMIC_PARAMS = params(version)

        winner = pop.run(self.eval_fitness_matches, gens)
        return winner, stats
    
    def eval_fitness_gen_i_train(self, genomes, config):
        pokemons = []
        mut_n = 0
        newborn_n = 0
        created_n = 0
        selected_n = 0
        self.age = self.age +1
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
                selected_n = selected_n +1
            else:
                if (genome.selected_parent[0] and genome.selected_parent[1]) and (genome.selected_parent[0] in self.genome_id_to_pokemon) and (get_U()<0.7):
                    parent: PokemonEntity = self.genome_id_to_pokemon[genome.selected_parent[0]]
                    pok = PokemonEntity(parent.index, genome, config)
                    pok.entity = parent.entity.copy()
                    pok.entity[0] = parent.entity[0]
                    pok.entity[0].moves = parent.entity[0].moves.copy()
                    pok.entity[0].nature = parent.entity[0].nature
                    pok.entity[0].ability = parent.entity[0].ability
                    newborn_n = newborn_n +1
                    if get_U() < 0.69 and genome.selected_parent[1] in self.genome_id_to_pokemon:
                        parent2: PokemonEntity = self.genome_id_to_pokemon[genome.selected_parent[1]]
                        self.crossover(pok, parent2)
                    pok.lvl = 1
                    pok.entity[0].level = 1
                    pok.xp = 0
                    self.genome_id_to_pokemon[genome_id] = pok
                    pokemons.append(pok)
                        
                else:
                    pok = PokemonEntity(self.index_generator(), genome, config)
                    self.genome_id_to_pokemon[genome_id] = pok
                    pokemons.append(pok)
                    created_n = created_n +1
            
            # if pok.genome.fitness>0.1:
            #     print("aaaa",pok.genome.fitness)
            
            U = get_U()
            p = pok.temperature()
            if U < p and pok.genome.fitness > 0.005:
                mut_n = mut_n + self.mutate(pok)
           
        print("Species:",pokemons[0].entity[0].species)
        print("Seleccionados:", selected_n,"pokemons de", len(pokemons))
        print("Nacidos:", newborn_n,"pokemons de", len(pokemons))
        print("Creados:", created_n,"pokemons de", len(pokemons))
        print("Mutados:", mut_n,"pokemons de", len(pokemons))
        
        max_fitness = 0.0
        # self.pkmn_logic.pokemon_bipartite_reg_matches(pokemons, 30)
        # print("iteration started")
        for i,pkmn in enumerate(pokemons):
            pkmn: PokemonEntity
            pkmn.used_moves = set()
            pkmn.age = pkmn.age +1
            
            if (np.mod(i,10)==0):
                print("pokemon", i)
            pkmn:PokemonEntity
            pkmn.won_battles = 0
            pkmn.total_battles = 0
            for _ in range(55):
                foe: PokemonEntity = np.random.choice(self.available_pokemons)
                foe.lvl = pkmn.lvl
                foe.entity[0].level = pkmn.lvl
                pkmn.won_battles = pkmn.won_battles +self.pkmn_logic.battle(pkmn, foe)
                pkmn.total_battles = pkmn.total_battles +1
            
            pkmn.update_fitness(self.age)
            if pkmn.genome.fitness > max_fitness: #(pkmn.lvl > 90) or (pkmn.won_battles > 46):
                max_fitness = pkmn.genome.fitness
                if max_fitness > self.m_fitness:
                    self.m_fitness = max_fitness
                    print()
                    print("pkmn index",i)
                    print("pkmn lvl", pkmn.lvl)
                    print("pkmn age", pkmn.age)
                    print("pkmn used moves", list(pkmn.used_moves))
                    print("battles won",pkmn.won_battles,"/",pkmn.total_battles)
                    print("pkmn fit", pkmn.genome.fitness)
                if self.ax_method:
                    self.ax_method(max_fitness)
                        
        # for pokemon in pokemons:
        #     pokemon:PokemonEntity
        #     pokemon.update_fitness()
            
    def run_i_gen_training(self, gens, version):
        pop = neat.population.Population(CONFIG)
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.reporting.StdOutReporter(True))

        global DYNAMIC_PARAMS
        DYNAMIC_PARAMS = params(version)

        winner = pop.run(self.eval_fitness_gen_i_train, gens)
        return winner, stats
    
def gen_i_training():
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
        genetic = GeneticEvolution(None)
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
        
        top_genome = genetic.run_i_gen_training(NO_OF_GENS, BRAIN_SIZE)[0]
        winner: PokemonEntity = genetic.genome_id_to_pokemon[top_genome.key]
        
        print()
        print("Results:")
        print("Specie:", winner.entity[0].species)
        print("Won:",winner.won_battles, "from", winner.total_battles)
        print("Level:",winner.lvl)
        print("Age:",winner.age)
        print("Used Moves:",list(winner.used_moves))
        print("Fitness:", winner.genome.fitness)
        print("Ability:", winner.entity[0].ability)
        print("Nature:", winner.entity[0].nature)
        print("Moves:", *winner.entity[0].moves, sep = "\n - ")
        print()
        # print("press a key")
        # input()
        save_pokemon(winner)
        print("pokemon_saved")
    
if __name__ == "__main__":
    gen_i_training()
    
    # pkmns = []
    # for pat in Path("./pokemons/").iterdir():
    #     pkmns.append(load_pokemon(pat))
    
    # BasicPkmnLogic().pokemon_match_old(pkmns)
        
    # pkmn1 = load_pokemon('pokemons/to_train/azelf_timid_small_100.pkl')
    # pkmn2 = load_pokemon('pokemons/charizard_bold_small_97.pkl')
    
    # pokemon_logic = BasicPkmnLogic()
    # pokemon_logic.battle(pkmn1, pkmn2)
    # for log in pokemon_logic.lb_log:
    #     print(log)
    #     print()
    #     input()
    # A.genome = None
    # A.lvl = 46
    # B.lvl = 46
    # A.entity[0].level = 50
    # B.entity[0].level = 50
        
        
    
    
    
    
    
    
        #     print("Specie:", pok.entity[0].species)
        #     print("Fitness:", pok.genome.fitness)
        #     print("Ability:", pok.entity[0].ability)
        #     print("Nature:", pok.entity[0].nature)
        #     print("Moves:", *pok.entity[0].moves, sep = "\n - ")
        #     print()
                
    # def gen():
    #     U = get_U()
    #     if U < 0.33333333:
    #         return 3
    #     elif U < 0.66666666:
    #         return 6
    #     else:
    #         return 9
    
    # genetic = GeneticEvolution(gen)
    # top_genome = genetic.run_matches_itself(NO_OF_GENS, BRAIN_SIZE)[0]
    # winner: PokemonEntity = genetic.genome_id_to_pokemon[top_genome.key]
    # print("Fitness:", winner.genome.fitness)
    # print("Specie:", winner.entity[0].species)
    # print("Ability:", winner.entity[0].ability)
    # print("Nature:", winner.entity[0].nature)
    # print("Moves:", *winner.entity[0].moves, sep = "\n - ")
    # save_pokemon(winner)
    
    # a = PokemonEntity(1,None,None, level= 80, has_ai=False)
    # print(a.entity[0].)
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
