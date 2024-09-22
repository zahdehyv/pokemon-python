import pickle
import neat
import neat.nn
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import sim.sim as sim
from tools.pick_six import generate_team_pok
from sim.structs import Pokemon
import numpy as np

COOLER = 3

BRAIN_SIZE = "S"
BRAIN_SIZE_TEXT = "small" if BRAIN_SIZE == "S" else "medium" if BRAIN_SIZE == "M" else "large"          

INPUT_COORDINATES = [(float(j-6), float(-2)) for j in range(13)]
OUTPUT_COORDINATES = [(float(x-3), 2.0) for x in list(range(3))+list(range(4,7))]

SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

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

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def sample_from_distribution(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)


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
        self.used_moves = {}
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
        # return net_output.index(max(net_output)) -1    
        return sample_from_distribution(softmax(net_output)) -1 
    
    def fitness(self, age):
        x = self.lvl/100
        if self.total_battles > 0:
            x = 0.4*x + 0.6*(self.won_battles/self.total_battles)
        p = (3/(age + 2.3))**2
        return x
        # return p*((len(self.used_moves)/4)**2.9) + (1-p)*x
    
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
