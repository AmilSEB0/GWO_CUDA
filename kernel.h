#ifndef KERNEL_GWO_H
#define KERNEL_GWO_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================
   PARAMÈTRES DU PROBLÈME
   ========================================================= */

/*
Bornes de l’espace de recherche.

Tous les loups (solutions) évolueront dans cet intervalle
pour chaque dimension.
*/
#define LOWER_BOUND -10.0
#define UPPER_BOUND  10.0


/* =========================================================
   STRUCTURES DE DONNÉES
   ========================================================= */

/*
Structure utilisée pour stocker les 3 meilleures solutions
(alpha, beta, delta) trouvées dans une population.

- val : valeurs de fitness (plus petit = meilleur)
- idx : indices des loups correspondants dans la population
*/
struct Best3 {
    double val[3];  // meilleures fitness
    int idx[3];     // indices des loups correspondants
};

/* =========================================================
   FONCTIONS DEVICE (GPU uniquement)
   ========================================================= */

/*
Fonction de fitness exécutée sur GPU.

Chaque thread peut évaluer une solution (un loup).

x                 : vecteur solution
max_dim           : nombre de dimensions
selected_obj_func : type de fonction objectif
*/
__device__ double device_fitness_function(
    const double* x,
    int max_dim,
    int selected_obj_func
);

/* =========================================================
   KERNELS CUDA (exécutés sur GPU)
   ========================================================= */

/*
Initialise les générateurs de nombres aléatoires (curand).

Chaque thread reçoit son propre état RNG pour éviter
les corrélations entre threads.
*/
__global__ void kernelInitRNG(
    curandState *states,
    unsigned long seed,
    int pop,
    int max_dim
);

/*
Initialise la population de loups avec des valeurs aléatoires
dans [LOWER_BOUND, UPPER_BOUND].

positions : tableau de taille (pop * dim)
*/
__global__ void kernelInitPopulation(
    double *positions,
    curandState *rngStates,
    int total
);

/*
Met à jour les positions des loups selon l’algorithme
Grey Wolf Optimizer (GWO).

Chaque loup se déplace en fonction de :
- alpha (meilleur)
- beta  (2e meilleur)
- delta (3e meilleur)

a : paramètre qui décroît au cours des itérations
    (contrôle exploration → exploitation)
*/
__global__ void kernelUpdatePositionsGWO(
    double *positions,
    double *alpha,
    double *beta,
    double *delta,
    double *new_positions,
    double a,
    curandState *rngStates,
    int pop,
    int dim
);

/*
Calcule la fitness de chaque loup.

1 thread = 1 loup
*/
__global__ void kernelComputeFitness(
    double *positions,
    double *fitness,
    int pop,
    int max_dim,
    int selected_obj_func
);

/*
Sélection élitiste :

Compare ancienne et nouvelle position d’un loup
et garde la meilleure (minimisation).

→ empêche la dégradation des solutions
*/
__global__ void kernelSelectBetter(
    double *oldPos,
    double *newPos,
    double *fitnessOld,
    double *fitnessNew,
    int pop,
    int max_dim
);

/*
Trouve les 3 meilleurs loups dans chaque bloc GPU.

Chaque bloc produit un Best3 (réduction locale).
*/
__global__ void kernelFindBest3(
    double *fitness,
    Best3 *blockBest,
    int pop
);

/*
Fusionne les résultats de tous les blocs pour obtenir
les 3 meilleurs globaux.

→ réduction finale
*/
__global__ void kernelReduceBest3(
    Best3 *input,
    Best3 *output,
    int n
);

/*
Extrait les vecteurs alpha, beta, delta à partir
des indices trouvés.

positions → alpha, beta, delta
*/
__global__ void extractEliteFromBest3(
    double *positions,
    double *alpha,
    double *beta,
    double *delta,
    Best3 *best,
    int dim
);


/* =========================================================
   FONCTION PRINCIPALE CUDA (appel CPU)
   ========================================================= */

/*
Fonction principale exécutant le Grey Wolf Optimizer sur GPU.

pop               : nombre de loups (population)
dim               : nombre de dimensions
max_epoch         : nombre d’itérations
selected_obj_func : fonction objectif (0,1,2)
seed              : graine aléatoire

Retour :
→ meilleure fitness trouvée
*/
double cuda_gwo(
    int pop,
    int dim,
    int max_epoch,
    int selected_obj_func,
    unsigned long seed
);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_GWO_H