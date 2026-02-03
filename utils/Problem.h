#ifndef PROBLEM_H
#define PROBLEM_H
#include <vector>
#include <functional>
#include <memory>  // Nécessaire pour std::shared_ptr

class Problem {
public:
    // Constructeur du problème d'optimisation
    // 'lb' : bornes inférieures pour chaque dimension
    // 'ub' : bornes supérieures pour chaque dimension
    // 'minmax' : chaîne indiquant si le problème est de minimisation ou de maximisation
    // 'obj_func' : fonction d'objectif qui évalue une solution donnée
    Problem(const std::vector<std::vector<double>>& lb, const std::vector<std::vector<double>>& ub, const std::string& minmax,
            const std::function<std::vector<double>(const std::vector<double>&)>& obj_func);

    // Méthode pour corriger une solution afin qu'elle respecte les bornes du problème
    std::vector<double> correct_solution(const std::vector<double>& x) const;

    // Génère une solution aléatoire dans les bornes définies
    std::vector<double> generate_solution();

    // Getter pour le nombre de dimensions du problème
    size_t getNDims() const;

    // Getter pour savoir si le problème est de minimisation ou de maximisation
    std::string getMinMax() const;

    // Calcule et retourne directement la fitness d'une solution donnée
    double get_target(const std::vector<double>& solution) const;

private:
    // Attributs définissant les bornes pour chaque dimension du problème
    std::vector<std::vector<double>> lb_;  // Liste des bornes inférieures pour chaque dimension
    std::vector<std::vector<double>> ub_;  // Liste des bornes supérieures pour chaque dimension

    // Vecteurs pour faciliter la gestion des bornes
    std::vector<double> lb_flat_;  // Vecteur des bornes inférieures
    std::vector<double> ub_flat_;  // Vecteur des bornes supérieures

    // Attribut pour définir si le problème est une minimisation ou une maximisation
    std::string minmax_;

    // Fonction d'objectif utilisée pour évaluer les solutions
    std::function<std::vector<double>(const std::vector<double>&)> obj_func_;

    // Poids associés aux objectifs (pour un problème multi-objectifs)
    std::vector<double> obj_weights_;

    // Nombre de dimensions du problème
    size_t n_dims_;

    // Méthodes internes pour configurer les bornes et les fonctions d'objectifs
    void set_bounds(const std::vector<std::vector<double>>& lb, const std::vector<std::vector<double>>& ub);
    void set_functions();
};

#endif // PROBLEM_H