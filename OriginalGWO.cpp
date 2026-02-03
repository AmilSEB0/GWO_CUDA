#include "OriginalGWO.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <map>

// Constructeur de la classe OriginalGWO
OriginalGWO::OriginalGWO(int epoch, int pop_size) : g_best(nullptr), problem(nullptr) {
    // Vérifie et assigne les valeurs pour epoch et pop_size
    this->epoch = checkInt("epoch", epoch, {1, 100000});
    this->pop_size = checkInt("pop_size", pop_size, {5, 10000});
}

// Destructeur de la classe
OriginalGWO::~OriginalGWO() {
    // Nettoyage des populations et des ressources utilisées
    pop.clear();
}

// Fonction pour vérifier et valider les valeurs entières dans les bornes spécifiées
double OriginalGWO::checkInt(const std::string& name, double value, std::pair<int, int> bound) {
    // Si la valeur est en dehors des bornes, une exception est lancée
    if (value < bound.first || value > bound.second) {
        throw std::invalid_argument(name + " must be in range [" + std::to_string(bound.first) + ", " + std::to_string(bound.second) + "].");
    }
    return static_cast<int>(value); // Retourne la valeur convertie en entier
}

// Fonction principale de résolution pour l'optimiseur GWO
std::shared_ptr<Agent> OriginalGWO::solve(Problem* problem) {
    if (!problem) {
        throw std::invalid_argument("Problem instance is null.");
    }
    this->problem = problem; // Assigne l'instance du problème

    // Génère la population initiale
    pop = generate_population(pop_size); // Crée la population initiale

    // Exécute l'optimisation pour plusieurs epochs
    for (int e = 1; e <= epoch; ++e) {
        evolve(e); // Met à jour la population selon l'algorithme GWO
        g_best = update_global_best_agent(pop); // Mise à jour de l'agent avec la meilleure performance
    }

    return g_best; // Retourne l'agent avec la meilleure performance
}

// Crée un agent vide avec la solution spécifiée
std::shared_ptr<Agent> OriginalGWO::generate_empty_agent(const std::vector<double>& solution) {
    std::vector<double> agent_solution = solution.empty() ? problem->generate_solution() : solution;
    return std::make_shared<Agent>(agent_solution, problem->get_target(agent_solution)); // Crée un agent avec la solution
}

// Crée un agent avec une solution donnée
std::shared_ptr<Agent> OriginalGWO::generate_agent(const std::vector<double>& solution) {
    auto agent = generate_empty_agent(solution);
    agent->set_target(get_target(agent->get_solution()));  // Assigne le fitness à l'agent
    return agent;
}

// Retourne l'agent ayant la meilleure performance globale (fitness)
std::shared_ptr<Agent> OriginalGWO::get_global_best() const {
    if (this->g_best == nullptr) {
        throw std::runtime_error("No global best found. The optimizer has not run yet.");
    }
    return this->g_best; // Retourne l'agent avec la meilleure performance
}

// Retourne l'agent ayant la meilleure performance entre deux agents
std::shared_ptr<Agent> OriginalGWO::get_better_agent(
    std::shared_ptr<Agent> agent_x,
    std::shared_ptr<Agent> agent_y,
    const std::string& minmax,
    bool reverse) {
    /**
     * Compare deux agents et retourne celui avec la meilleure performance selon le type de "fitness".
     */
    if (agent_y == nullptr) {
        return agent_x;  // Si l'un des agents est nul, retourne l'autre
    }
    std::map<std::string, int> minmax_dict = {{"min", 0}, {"max", 1}};
    int idx = minmax_dict[minmax];  // 0 pour minimiser, 1 pour maximiser

    if (reverse) {
        idx = 1 - idx;  // Inverse la comparaison si nécessaire
    }

    // Compare les agents en fonction de leur "fitness"
    if (idx == 0) {  // Min (Cherche la meilleure performance "fitness" minimale)
        return (agent_x->get_target() < agent_y->get_target()) ? agent_x : agent_y;
    } else {  // Max (Cherche la meilleure performance "fitness" maximale)
        return (agent_x->get_target() < agent_y->get_target()) ? agent_y : agent_x;
    }
}

// Génère une population d'agents
std::vector<std::shared_ptr<Agent>> OriginalGWO::generate_population(int pop_size) {
    if (pop_size == 0) {
        pop_size = this->pop_size; // Utilise la taille par défaut si la taille est nulle
    }

    std::vector<std::shared_ptr<Agent>> population;
    for (int i = 0; i < pop_size; ++i) {
        population.push_back(generate_agent()); // Crée un agent
    }

    return population; // Retourne la population
}

// Corrige la solution selon les contraintes du problème
std::vector<double> OriginalGWO::correct_solution(const std::vector<double>& solution) {
    return problem->correct_solution(solution); // Applique les corrections nécessaires à la solution
}

// Retourne le fitness (cible) d'une solution donnée
double OriginalGWO::get_target(const std::vector<double>& solution) {
    return problem->get_target(solution); // Appelle la méthode de fitness du problème
}

// Met à jour l'agent globalement meilleur dans la population
std::shared_ptr<Agent> OriginalGWO::update_global_best_agent(std::vector<std::shared_ptr<Agent>>& pop) {
    std::vector<double> list_fits;
    // Collecte les valeurs de fitness des agents
    for (auto& agent : pop) {
        list_fits.push_back(agent->get_target());
    }

    std::vector<int> indices(pop.size());
    std::iota(indices.begin(), indices.end(), 0); // Initialise les indices

    // Trie les indices en fonction des valeurs de fitness
    std::sort(indices.begin(), indices.end(), [&list_fits](int a, int b) {
        return list_fits[a] < list_fits[b];
    });

    // Inverse si l'optimisation est en maximisation
    if (this->problem->getMinMax() == "max") {
        std::reverse(indices.begin(), indices.end());
    }

    std::vector<std::shared_ptr<Agent>> sorted_pop;
    for (int idx : indices) {
        sorted_pop.push_back(pop[idx]);
    }

    auto c_best = sorted_pop[0];  // Agent avec la meilleure performance (fitness)

    // Sauvegarde du meilleur agent trouvé
    auto better = get_better_agent(c_best, g_best, this->problem->getMinMax());

    return better;  // Retourne l'agent avec la meilleure performance
}

// Génère un vecteur de nombres aléatoires dans un intervalle donné
std::vector<double> OriginalGWO::generate_random_vector(int size, double min, double max) {
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = min + (max - min) * ((double) rand() / RAND_MAX); // Génère une valeur aléatoire entre min et max
    }
    return vec;
}

void OriginalGWO::evolve(int epoch) {
    // Décroissance linéaire de la valeur de a de 2 à 0
    double a = 2.0 - 2.0 * epoch / this->epoch;

    // Trie de la population en fonction de la fitness
    std::vector<double> list_fits;
    for (auto& agent : pop) {
        list_fits.push_back(agent->get_target());
    }

    std::vector<int> indices(list_fits.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&list_fits](int i1, int i2) {
        return list_fits[i1] < list_fits[i2]; // Trie selon la fitness
    });

    if (problem->getMinMax() == "max") {
        std::reverse(indices.begin(), indices.end()); // Inverse si l'optimisation est en maximisation
    }

    // Recrée la population triée avec des shared_ptr
    std::vector<std::shared_ptr<Agent>> sorted_pop;
    for (auto idx : indices) {
        sorted_pop.push_back(pop[idx]);
    }
    pop = sorted_pop;

    // Sélectionne les 3 meilleurs agents
    std::vector<std::shared_ptr<Agent>> list_best(pop.begin(), pop.begin() + 3);

    // Initialisation de la nouvelle population
    std::vector<std::shared_ptr<Agent>> pop_new;
    // Crée des nouveaux agents pour la population
    for (int idx = 0; idx < pop_size; ++idx) {
        // Génère des vecteurs aléatoires pour A1, A2, A3, C1, C2, C3
        std::vector<double> A1 = generate_random_vector(problem->getNDims(), -1.0, 1.0);
        for (auto& val : A1) {
            val *= a;  // Multiplication élément par élément
        }

        std::vector<double> A2 = generate_random_vector(problem->getNDims(), -1.0, 1.0);
        for (auto& val : A2) {
            val *= a;
        }

        std::vector<double> A3 = generate_random_vector(problem->getNDims(), -1.0, 1.0);
        for (auto& val : A3) {
            val *= a;
        }

        std::vector<double> C1 = generate_random_vector(problem->getNDims(), 0.0, 2.0);
        std::vector<double> C2 = generate_random_vector(problem->getNDims(), 0.0, 2.0);
        std::vector<double> C3 = generate_random_vector(problem->getNDims(), 0.0, 2.0);

        // Calcul des nouvelles positions des agents
        std::vector<double> X1 = list_best[0]->get_solution(); // Alpha
        std::vector<double> X2 = list_best[1]->get_solution(); // Beta
        std::vector<double> X3 = list_best[2]->get_solution(); // Delta

        // Mise à jour des positions X1, X2, X3 selon les formules du GWO
        for (size_t i = 0; i < X1.size(); ++i) {
            X1[i] -= A1[i] * std::abs(C1[i] * X1[i] - pop[idx]->get_solution()[i]);
            X2[i] -= A2[i] * std::abs(C2[i] * X2[i] - pop[idx]->get_solution()[i]);
            X3[i] -= A3[i] * std::abs(C3[i] * X3[i] - pop[idx]->get_solution()[i]);
        }

        // Calcul de la position moyenne
        std::vector<double> pos_new(X1.size());
        for (size_t i = 0; i < X1.size(); ++i) {
            pos_new[i] = (X1[i] + X2[i] + X3[i]) / 3.0;
        }

        // Correction de la position
        pos_new = correct_solution(pos_new);

        // Génère un agent avec la nouvelle position
        auto agent = generate_empty_agent(pos_new);
        pop_new.push_back(agent);

        // Mise à jour du fitness de l'agent
        agent->set_target(get_target(pos_new));

        // Mise à jour de la population avec l'agent amélioré
        pop[idx] = get_better_agent(agent, pop[idx], problem->getMinMax());
    }

    // Mise à jour de la population
    pop = pop_new;
}