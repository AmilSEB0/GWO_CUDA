
/*
=========================================================
PROGRAMME PRINCIPAL (CPU)
=========================================================

Rôle :
- Tester l’algorithme CUDA GWO
- Lancer plusieurs expériences
- Sauvegarder les résultats dans des fichiers CSV
- Calculer des statistiques (moyenne, écart-type)

Sorties :
1. execution_resultssansprint.csv → résultats de chaque run
2. mean_std_fitnesssansprint.csv → statistiques globales
*/

#include "kernel.h"      // Contient cuda_gwo et constantes
#include <ctime>         // Pour mesurer le temps
#include <cstdio>        // Entrées/sorties C
#include <cstdlib>       // Fonctions utilitaires (rand, srand)
#include <vector>        // Tableaux dynamiques
#include <cuda_runtime.h> // Synchronisation GPU
#include <iostream>      // Affichage console
#include <fstream>       // Fichiers (CSV)
#include <string>        // Gestion chaînes
#include <cfloat>        // Constantes float
#include <iomanip>       // Format d'affichage


// =======================================================
// NOM DES FONCTIONS OBJECTIF
// =======================================================

/*
Convertit un identifiant numérique en nom de fonction
(utile pour affichage et CSV)
*/
std::string obj_func(int selected_obj_func)
{
    switch (selected_obj_func)
    {
    case 0: return "Rastrigin";   // Fonction multimodale
    case 1: return "Rosenbrock";  // Vallée difficile
    case 2: return "Ackley";      // Très complexe
    default: return "Unknown";    // Cas erreur
    }
}


// =======================================================
// PROGRAMME PRINCIPAL
// =======================================================

int main(int argc, char** argv)
{
    // ===================================================
    // FICHIERS CSV
    // ===================================================

    /*
    Fichier 1 :
    Contient les résultats de CHAQUE exécution
    */
    std::ofstream execution_file("execution_results.csv");

    // Format numérique précis (important pour analyse scientifique)
    execution_file << std::defaultfloat << std::setprecision(15);

    // En-tête CSV
    execution_file << "Function,Dimension,Population,Run,Fitness,Execution_Time\n";


    /*
    Fichier 2 :
    Contient les statistiques globales (moyenne + écart-type)
    */
    std::ofstream stats_file("mean_std_fitness.csv");

    stats_file << std::defaultfloat << std::setprecision(15);

    stats_file << "Function,Dimension,Population,MeanFitness,StdFitness,MeanTime,StdTime\n";


    // ===================================================
    // PARAMÈTRES DES EXPÉRIENCES
    // ===================================================

    /*
    Différentes tailles de population à tester
    */
    std::vector<int> pop_sizes = {50, 100, 500};

    /*
    Différentes dimensions du problème
    */
    std::vector<int> dims = {10, 50, 100};

    /*
    Fonctions objectif testées
    */
    std::vector<int> selected_obj_funcs = {0, 1, 2};

    /*
    Nombre de répétitions pour chaque configuration
    */
    const int RUNS = 10;

    // Initialisation du générateur aléatoire CPU
    srand((unsigned)time(NULL));


    // ===================================================
    // BOUCLES PRINCIPALES (expérimentations)
    // ===================================================

    /*
    On teste toutes les combinaisons :
    population × dimension × fonction
    */
    for (int pop_size : pop_sizes)
    {
        for (int dim : dims)
        {
            for (int selected_obj_func : selected_obj_funcs)
            {
                /*
                Calcul du nombre d’itérations :
                → dépend de la dimension et population
                → maintient un coût global similaire
                */
                int max_epoch = (10000 * dim) / pop_size - 1;

                // Nom de la fonction (pour CSV)
                std::string function_name = obj_func(selected_obj_func);

                /*
                Tableaux pour stocker les résultats
                de chaque run
                */
                std::vector<double> results(RUNS);
                std::vector<double> times(RUNS);


                // ===================================================
                // BOUCLE DES RUNS
                // ===================================================

                for (int run = 0; run < RUNS; run++)
                {
                    // Début du chronométrage CPU
                    clock_t begin = clock();

                    /*
                    Seed différente à chaque run
                    → évite résultats identiques
                    */
                    unsigned long seed = (unsigned long)time(NULL) + run;

                    /*
                    APPEL PRINCIPAL GPU

                    cuda_gwo :
                    → exécute l’algorithme GWO sur GPU
                    → retourne la meilleure fitness trouvée
                    */
                    double bestFitness = cuda_gwo(
                        pop_size,
                        dim,
                        max_epoch,
                        selected_obj_func,
                        seed
                    );

                    /*
                    Synchronisation GPU
                    → s’assurer que tous les calculs sont terminés
                    avant de mesurer le temps
                    */
                    cudaDeviceSynchronize();

                    // Fin du chronométrage
                    clock_t end = clock();

                    /*
                    Temps d’exécution en secondes
                    */
                    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

                    // Sauvegarde des résultats
                    results[run] = bestFitness;
                    times[run] = time_spent;

                    // ===================================================
                    // ÉCRITURE CSV (résultats individuels)
                    // ===================================================

                    execution_file << function_name << ","
                                   << dim << ","
                                   << pop_size << ","
                                   << run + 1 << ","
                                   << bestFitness << ","
                                   << time_spent << "\n";
                }


                // ===================================================
                // CALCUL DES STATISTIQUES
                // ===================================================

                double sum = 0.0, sumTime = 0.0;

                // Calcul des sommes
                for (int i = 0; i < RUNS; i++)
                {
                    sum += results[i];
                    sumTime += times[i];
                }

                // Moyennes
                double mean = sum / RUNS;
                double meanTime = sumTime / RUNS;

                double var = 0.0, varTime = 0.0;

                // Calcul variance
                for (int i = 0; i < RUNS; i++)
                {
                    var += pow(results[i] - mean, 2);
                    varTime += pow(times[i] - meanTime, 2);
                }

                // Écart-type
                double stddev = sqrt(var / RUNS);
                double stdTime = sqrt(varTime / RUNS);


                // ===================================================
                // ÉCRITURE CSV (statistiques)
                // ===================================================

                stats_file << function_name << ","
                           << dim << ","
                           << pop_size << ","
                           << mean << ","
                           << stddev << ","
                           << meanTime << ","
                           << stdTime << "\n";
            }
        }
    }


    // ===================================================
    // FERMETURE DES FICHIERS
    // ===================================================

    execution_file.close(); // ferme fichier runs
    stats_file.close();     // ferme fichier stats

    // Fin du programme
    return 0;
}