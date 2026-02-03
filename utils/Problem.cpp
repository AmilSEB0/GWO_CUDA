#include "Problem.h"
#include <stdexcept>
#include <random>

Problem::Problem(const std::vector<std::vector<double>>& lb, const std::vector<std::vector<double>>& ub, const std::string& minmax,
                 const std::function<std::vector<double>(const std::vector<double>&)>& obj_func)
    : lb_(lb), ub_(ub), minmax_(minmax), obj_func_(obj_func) {
    set_bounds(lb, ub);  // Initialisation des bornes du problème
    set_functions();  // Initialisation des fonctions d'objectifs
}

// Fonction pour définir les bornes du problème
void Problem::set_bounds(const std::vector<std::vector<double>>& lb, const std::vector<std::vector<double>>& ub) {
    lb_ = lb;  // Enregistre les bornes inférieures
    ub_ = ub;  // Enregistre les bornes supérieures

    lb_flat_.clear();  // Réinitialisation du vecteur pour les bornes inférieures
    ub_flat_.clear();  // Réinitialisation du vecteur pour les bornes supérieures

    for (size_t i = 0; i < lb_.size(); ++i) {
        lb_flat_.insert(lb_flat_.end(), lb_[i].begin(), lb_[i].end());  // Fusion des bornes inférieures
        ub_flat_.insert(ub_flat_.end(), ub_[i].begin(), ub_[i].end());  // Fusion des bornes supérieures
    }
}

// Fonction pour corriger une solution (en s'assurant qu'elle respecte les bornes)
std::vector<double> Problem::correct_solution(const std::vector<double>& x) const {
    std::vector<double> x_new;  // Vecteur pour stocker la solution corrigée
    size_t n_vars = 0;  // Compteur pour les variables

    // Correction de chaque variable selon ses bornes
    for (size_t i = 0; i < lb_.size(); ++i) {
        for (size_t j = 0; j < lb_[i].size(); ++j) {
            // Clamping de la valeur dans l'intervalle [lb, ub]
            double corrected_value = std::clamp(x[n_vars + j], lb_[i][j], ub_[i][j]);
            x_new.push_back(corrected_value);
        }
        n_vars += lb_[i].size();  // Mise à jour du compteur de variables
    }

    return x_new; // Retourne la solution corrigée
}

// Fonction pour générer une solution aléatoire dans les bornes
std::vector<double> Problem::generate_solution() {
    std::vector<double> x;  // Vecteur pour stocker la solution générée
    std::random_device rd;  // Initialisation de la génération aléatoire
    std::mt19937 gen(rd());  // Générateur de nombres aléatoires

    // Génération des variables dans les bornes spécifiées
    for (size_t i = 0; i < lb_.size(); ++i) {
        for (size_t j = 0; j < lb_[i].size(); ++j) {
            std::uniform_real_distribution<> dis(lb_[i][j], ub_[i][j]);  // Distribution uniforme pour chaque variable
            x.push_back(dis(gen));  // Ajout de la variable générée à la solution
        }
    }

    return x;  // Retourne la solution générée
}

// Fonction pour initialiser les fonctions d'objectifs et les poids associés
void Problem::set_functions() {
    std::vector<double> tested_solution = generate_solution();  // Génère une solution de test
    n_dims_ = tested_solution.size();  // Nombre de dimensions de la solution
    std::vector<double> result = obj_func_(tested_solution);  // Evaluation de la solution par la fonction d'objectif

    // Vérifie que la fonction d'objectif retourne des résultats valides
    if (result.empty()) {
        throw std::invalid_argument("obj_func doit retourner un vecteur non vide de valeurs.");
    }

    obj_weights_ = std::vector<double>(result.size(), 1.0);  // Initialisation des poids des objectifs (égal à 1 pour chaque objectif)
}

// Getter pour le nombre de dimensions de la solution
size_t Problem::getNDims() const {
    return n_dims_;
}

// Getter pour savoir si le problème est un problème de minimisation ou de maximisation
std::string Problem::getMinMax() const {
    return minmax_;
}

// Fonction pour calculer la fitness d'une solution
double Problem::get_target(const std::vector<double>& solution) const {
    std::vector<double> objectives = obj_func_(solution);  // Evaluation de la solution par la fonction d'objectif
    double fitness = 0.0;  // Initialisation de la fitness

    // Calcul de la fitness en fonction des objectifs et des poids
    for (size_t i = 0; i < objectives.size(); ++i) {
        fitness += objectives[i] * obj_weights_[i];  // Calcul du produit scalaire entre les objectifs et leurs poids
    }

    return fitness;  // Retourne la fitness calculée
}