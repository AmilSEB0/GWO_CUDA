#include "Agent.h"
#include <memory>  // Inclusion nécessaire pour utiliser std::shared_ptr

// Constructeur avec une solution et une valeur de fitness (target)
Agent::Agent(const std::vector<double>& solution, double target)
    : solution(solution), target(target) {}  // Initialisation de la solution et de la target

// Constructeur par défaut
Agent::Agent() : target(0.0) {}  // La target est initialisée à 0.0, la solution est vide par défaut

// Méthode de copie
std::shared_ptr<Agent> Agent::copy() const {
    // Crée et retourne un shared_ptr pointant vers une copie de l'instance actuelle
    return std::make_shared<Agent>(*this);  // Utilisation de make_shared pour gérer la mémoire automatiquement
}

// Getters
std::vector<double> Agent::get_solution() const {
    return solution;  // Retourne la solution de l'agent sous forme de vecteur
}

double Agent::get_target() const {
    return target;  // Retourne la valeur de fitness (target) de l'agent
}

// Setters
void Agent::set_target(double target) {
    this->target = target;  // Permet de mettre à jour la valeur de la target de l'agent
}