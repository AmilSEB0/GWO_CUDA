#ifndef ORIGINAL_GWO_H
#define ORIGINAL_GWO_H

#include <vector>
#include <memory>
#include "utils/Agent.h"
#include "utils/Problem.h"
#include <random>

class OriginalGWO {
public:
    OriginalGWO(int epoch = 10000, int pop_size = 100);

    // Fonction qui évolue la population d'agents pendant un nombre d'époques donné
    void evolve(int epoch);

    // Destructeur de la classe OriginalGWO
    ~OriginalGWO();

    // Génère un vecteur de nombres aléatoires de taille "size" avec des valeurs comprises entre "min" et "max"
    std::vector<double> generate_random_vector(int size, double min, double max);

    // Méthode principale pour résoudre le problème d'optimisation en utilisant GWO
    // Retourne le meilleur agent trouvé
    std::shared_ptr<Agent> solve(Problem* problem);

    // Retourne l'agent avec la meilleure solution globale trouvée
    std::shared_ptr<Agent> get_global_best() const;

private:
    // Génère un agent vide avec une solution donnée
    std::shared_ptr<Agent> generate_empty_agent(const std::vector<double>& solution = {});

    // Génère un agent à partir d'une solution donnée
    std::shared_ptr<Agent> generate_agent(const std::vector<double>& solution = {});

    // Génère une population d'agents de taille "pop_size"
    std::vector<std::shared_ptr<Agent>> generate_population(int pop_size = 0);

    // Met à jour l'agent globalement meilleur à partir de la population actuelle
    std::shared_ptr<Agent> update_global_best_agent(std::vector<std::shared_ptr<Agent>>& pop);

    // Corrige une solution pour qu'elle respecte les contraintes du problème
    std::vector<double> correct_solution(const std::vector<double>& solution);

    // Calcule la "fitness" (target) d'une solution donnée
    double get_target(const std::vector<double>& solution);

    // Retourne l'agent avec la meilleure performance entre deux agents
    static std::shared_ptr<Agent> get_better_agent(std::shared_ptr<Agent> agent_x, std::shared_ptr<Agent> agent_y, const std::string& minmax = "min", bool reverse = false);

    // Liste des agents représentant la population
    std::vector<std::shared_ptr<Agent>> pop;

    // Pointeur vers le problème à résoudre
    Problem* problem;

    // Nombre d'époques (itérations) pour lesquelles l'optimisation doit être effectuée
    int epoch;

    // Taille de la population (nombre d'agents dans la population)
    int pop_size;

    // Générateur de nombres aléatoires (utilisé pour générer des solutions aléatoires)
    std::mt19937 generator;

    // Fonction pour vérifier et ajuster les valeurs entières (comme epoch et pop_size)
    double checkInt(const std::string& name, double value, std::pair<int, int> bound = {1, 100000});

    // Agent globalement meilleur trouvé au cours de l'optimisation
    std::shared_ptr<Agent> g_best;
};

#endif // ORIGINAL_GWO_H
