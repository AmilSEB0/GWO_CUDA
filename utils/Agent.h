#ifndef AGENT_H
#define AGENT_H

#include <vector>
#include <memory>  // Inclusion pour std::shared_ptr

class Agent {
public:
    // Constructeur avec une solution et une valeur de fitness
    // 'solution' : vecteur représentant la solution de l'agent
    // 'target' : évaluation de la solution, généralement une valeur numérique
    Agent(const std::vector<double>& solution, double target);

    // Constructeur par défaut
    // Initialise la target à 0 et la solution à vide
    Agent();

    // Méthode de copie
    // Crée et retourne un shared_ptr vers une copie de l'instance actuelle de l'agent
    std::shared_ptr<Agent> copy() const;

    // Getters
    // Retourne la solution de l'agent sous forme d'un vecteur de doubles
    std::vector<double> get_solution() const;

    // Retourne la valeur de la fitness de l'agent
    double get_target() const;

    // Setters
    // Permet de définir la valeur de fitness de l'agent
    void set_target(double fitness);

private:
    std::vector<double> solution;  // Contient la solution de l'agent
    double target;  // Évaluation de la fitness de l'agent
};

#endif // AGENT_H