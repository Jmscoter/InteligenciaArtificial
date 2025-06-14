import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def load_knapsack_data(filename):
    df = pd.read_csv(filename)

    capacity = int(df.iloc[-1]['valor'])
    items_df = df.iloc[:-1]
    weights = items_df['peso'].tolist()
    values = items_df['valor'].tolist()
    return weights, values, capacity


def calculate_fitness(chromosome, weights, values, capacity, penalty_factor=10):
    total_weight = sum(w for i, w in enumerate(weights) if chromosome[i] == 1)
    total_value = sum(v for i, v in enumerate(values) if chromosome[i] == 1)

    if total_weight > capacity:
        
        return 0 
    else:
        return total_value


def initialize_population_random(pop_size, num_items):
    population = []
    for _ in range(pop_size):
        chromosome = [random.randint(0, 1) for _ in range(num_items)]
        population.append(chromosome)
    return population

def initialize_population_heuristic_greedy(pop_size, num_items, weights, values, capacity):
    
    ratios = [(values[i] / weights[i], i) for i in range(num_items) if weights[i] > 0]
    ratios.sort(key=lambda x: x[0], reverse=True) 
    population = []
    for _ in range(pop_size): 
        chromosome = [0] * num_items
        current_weight = 0
        current_value = 0
        
        items_selected_greedy = []
        for ratio, i in ratios:
            if current_weight + weights[i] <= capacity:
                chromosome[i] = 1
                current_weight += weights[i]
                current_value += values[i]
                items_selected_greedy.append(i)
        
        
        for _ in range(random.randint(0, max(1, num_items // 10))):  
            idx = random.randint(0, num_items - 1)
            chromosome[idx] = 1 - chromosome[idx] 

        
        temp_weight = sum(w for i, w in enumerate(weights) if chromosome[i] == 1)
        if temp_weight > capacity:
            
            invalid_indices = [i for i, bit in enumerate(chromosome) if bit == 1]
            random.shuffle(invalid_indices)
            for idx_to_remove in invalid_indices:
                if sum(w for i, w in enumerate(weights) if chromosome[i] == 1) > capacity:
                    chromosome[idx_to_remove] = 0
                else:
                    break
        
        population.append(chromosome)
    return population



def select_parents_tournament(population, fitness_scores, tournament_size=3):
    selected_parents = []
    for _ in range(2): 
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_participants = [(fitness_scores[i], population[i]) for i in tournament_indices]
        winner = max(tournament_participants, key=lambda x: x[0])[1]
        selected_parents.append(winner)
    return selected_parents[0], selected_parents[1]


def crossover_one_point(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def crossover_two_points(parent1, parent2):
    p1, p2 = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child1, child2

def crossover_uniform(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5: 
            child1.append(parent1[i])
            child2.append(parent2[i])
        else: 
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2


def mutate(chromosome, mutation_rate):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i] 
    return mutated_chromosome


def run_genetic_algorithm(weights, values, capacity,
                          pop_size,
                          num_generations,
                          crossover_type,
                          mutation_rate,
                          initialization_type,
                          stop_criterion,
                          convergence_threshold=50): 
    
    num_items = len(weights)
    
    if initialization_type == "random":
        population = initialize_population_random(pop_size, num_items)
    elif initialization_type == "heuristic":
        population = initialize_population_heuristic_greedy(pop_size, num_items, weights, values, capacity)
    else:
        raise ValueError("Tipo de inicialização inválido. Use 'random' ou 'heuristic'.")

    best_overall_solution = None
    best_overall_fitness = -1
    
    fitness_history = [] 
    generations_without_improvement = 0

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(c, weights, values, capacity) for c in population]
        
        current_best_fitness = max(fitness_scores)
        current_best_solution = population[np.argmax(fitness_scores)]
        
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_solution = current_best_solution
            generations_without_improvement = 0 #
        else:
            generations_without_improvement += 1

        fitness_history.append(best_overall_fitness) 

       
        if stop_criterion == "convergence" and generations_without_improvement >= convergence_threshold:
            print(f"Parada por convergência na geração {generation}. Melhor fitness: {best_overall_fitness}")
            break

        new_population = []
        
        
        best_current_chromosome_idx = np.argmax(fitness_scores)
        new_population.append(population[best_current_chromosome_idx])

       
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents_tournament(population, fitness_scores)
            
            
            if crossover_type == "one_point":
                child1, child2 = crossover_one_point(parent1, parent2)
            elif crossover_type == "two_points":
                child1, child2 = crossover_two_points(parent1, parent2)
            elif crossover_type == "uniform":
                child1, child2 = crossover_uniform(parent1, parent2)
            else:
                raise ValueError("Tipo de crossover inválido.")
            
          
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
         
            new_population.append(child1)
            if len(new_population) < pop_size: 
                new_population.append(child2)
        
        population = new_population

    return best_overall_solution, best_overall_fitness, fitness_history


def run_experiments(knapsack_data_file, configs, num_runs_per_config=5):
    weights, values, capacity = load_knapsack_data(knapsack_data_file)
    num_items = len(weights)
    
    results = [] 
    
    for config_name, config_params in configs.items():
        print(f"\n--- Rodando experimento para: {config_name} ---")
        config_best_fitnesses = []
        config_fitness_histories = []
        
        for run in range(num_runs_per_config):
            print(f"  Rodada {run + 1}/{num_runs_per_config}...")
            best_solution, best_fitness, fitness_history = run_genetic_algorithm(
                weights, values, capacity,
                pop_size=config_params['pop_size'],
                num_generations=config_params['num_generations'],
                crossover_type=config_params['crossover_type'],
                mutation_rate=config_params['mutation_rate'],
                initialization_type=config_params['initialization_type'],
                stop_criterion=config_params['stop_criterion']
            )
            config_best_fitnesses.append(best_fitness)
            config_fitness_histories.append(fitness_history)
            
            print(f"    Melhor Fitness na Rodada {run+1}: {best_fitness}")
        
        avg_best_fitness = np.mean(config_best_fitnesses)
        std_best_fitness = np.std(config_best_fitnesses)
        
        results.append({
            'config_name': config_name,
            'params': config_params,
            'avg_best_fitness': avg_best_fitness,
            'std_best_fitness': std_best_fitness,
            'all_best_fitnesses': config_best_fitnesses,
            'all_fitness_histories': config_fitness_histories
        })
        print(f"Configuração {config_name} - Média da Melhor Fitness: {avg_best_fitness:.2f} (Std: {std_best_fitness:.2f})")
    
    return results


if __name__ == "__main__":
    
    knapsack_file = 'knapsack_4.csv' 


    test_configs = {
        "Config_1_OP_MutBaixa_Rand_FixedGen": {
            'pop_size': 100,
            'num_generations': 200,
            'crossover_type': 'one_point',
            'mutation_rate': 0.005, 
            'initialization_type': 'random',
            'stop_criterion': 'fixed_generations'
        },
        "Config_2_2P_MutMedia_Rand_FixedGen": {
            'pop_size': 100,
            'num_generations': 200,
            'crossover_type': 'two_points',
            'mutation_rate': 0.01, 
            'initialization_type': 'random',
            'stop_criterion': 'fixed_generations'
        },
        "Config_3_Uniform_MutAlta_Rand_FixedGen": {
            'pop_size': 100,
            'num_generations': 200,
            'crossover_type': 'uniform',
            'mutation_rate': 0.05,
            'initialization_type': 'random',
            'stop_criterion': 'fixed_generations'
        },
        "Config_4_OP_MutBaixa_Heuristic_FixedGen": {
            'pop_size': 100,
            'num_generations': 200,
            'crossover_type': 'one_point',
            'mutation_rate': 0.005,
            'initialization_type': 'heuristic', 
            'stop_criterion': 'fixed_generations'
        },
        "Config_5_OP_MutBaixa_Rand_Convergence": {
            'pop_size': 100,
            'num_generations': 500, 
            'crossover_type': 'one_point',
            'mutation_rate': 0.005,
            'initialization_type': 'random',
            'stop_criterion': 'convergence' 
        },
       
    }

    experiment_results = run_experiments(knapsack_file, test_configs, num_runs_per_config=5)

   
    print("\n--- Sumário dos Resultados dos Experimentos ---")
    for res in experiment_results:
        print(f"\nConfiguração: {res['config_name']}")
        print(f"  Parâmetros: {res['params']}")
        print(f"  Média da Melhor Fitness: {res['avg_best_fitness']:.2f}")
        print(f"  Desvio Padrão da Melhor Fitness: {res['std_best_fitness']:.2f}")

        
        plt.figure(figsize=(10, 6))
        for i, history in enumerate(res['all_fitness_histories']):
            plt.plot(history, label=f'Rodada {i+1}')
        plt.title(f'Evolução da Melhor Fitness para {res["config_name"]}')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.legend()
        plt.grid(True)
        plt.show()