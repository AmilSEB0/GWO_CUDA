/*
=======================================================
Implémentation CUDA du Grey Wolf Optimizer (GWO)
=======================================================

Principe :
- Chaque "loup" = une solution
- On minimise une fonction (Rastrigin, Rosenbrock, Ackley)
- Les 3 meilleurs loups guident les autres :
    alpha (meilleur)
    beta  (2e)
    delta (3e)
*/
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include "kernel.h"
#include <curand_kernel.h>
#include <ctime>

// =======================================================
// FONCTION DE FITNESS (GPU)
// =======================================================
/*
Cette fonction évalue la qualité d’une solution x.

- x : vecteur de dimension max_dim
- selected_obj_func :
    0 = Rastrigin (multi-minima)
    1 = Rosenbrock (vallée)
    2 = Ackley (très multimodale)

Retour :
→ valeur à MINIMISER
*/
__device__ double device_fitness_function(const double* x, int max_dim, int selected_obj_func)
{
    double res = 0.0;
    double PI = 3.14159265358979323846264338327950288;

    switch (selected_obj_func)
    {
        case 0: { // RASTRIGIN
            const double A = 10.0;
            for (int i = 0; i < max_dim; i++) {
                double zi = x[i];
                res += zi * zi - A * cos(2.0 * PI * zi);
            }
            res += A * max_dim;
            break;
        }

        case 1: { // ROSENBROCK
            for (int i = 0; i < max_dim - 1; i++) {
                double zi = x[i];
                double zip1 = x[i + 1];
                double t1 = zip1 - zi * zi;
                double t2 = 1.0 - zi;
                res += 100.0 * t1 * t1 + t2 * t2;
            }
            break;
        }

        case 2: { // ACKLEY
            const double a = 20.0;
            const double b = 0.2;
            const double c = 2.0 * PI;

            double sum1 = 0.0;
            double sum2 = 0.0;

            for (int i = 0; i < max_dim; i++) {
                double zi = x[i];
                sum1 += zi * zi;
                sum2 += cos(c * zi);
            }

            res = -a * exp(-b * sqrt(sum1 / max_dim))
                  - exp(sum2 / max_dim)
                  + a + exp(1.0);
            break;
        }
    }
    return res;
}

// =======================================================
// INITIALISATION RNG (curand)
// =======================================================
/*
Chaque thread initialise son propre générateur aléatoire.

Afin d'éviter corrélation entre threads GPU
*/
__global__ void kernelInitRNG(curandState *states, unsigned long seed, int pop, int max_dim)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Chaque dimension de chaque loup a son RNG
    if (id < pop * max_dim)
        curand_init(seed, id, 0, &states[id]);
}

// =======================================================
// MISE À JOUR DES POSITIONS (GWO)
// =======================================================
/*
Chaque thread met à jour UNE dimension d’un loup.

Formule GWO :
- attraction vers alpha, beta, delta
- pondération aléatoire (A, C)
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
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pop * dim;

    if (idx >= total) return;

    // Dimension courante
    int d = idx % dim;

    // RNG local (plus rapide)
    curandState localState = rngStates[idx];

    // Tirages aléatoires
    double r1 = curand_uniform_double(&localState);
    double r2 = curand_uniform_double(&localState);
    double r3 = curand_uniform_double(&localState);
    double r4 = curand_uniform_double(&localState);
    double r5 = curand_uniform_double(&localState);
    double r6 = curand_uniform_double(&localState);

    // Coefficients GWO
    double A1 = a * (2.0 * r1 - 1.0);
    double A2 = a * (2.0 * r2 - 1.0);
    double A3 = a * (2.0 * r3 - 1.0);

    double C1 = 2.0 * r4;
    double C2 = 2.0 * r5;
    double C3 = 2.0 * r6;

    double X = positions[idx];

    // Influence des leaders
    double X1 = alpha[d] - A1 * fabs(C1 * alpha[d] - X);
    double X2 = beta[d]  - A2 * fabs(C2 * beta[d]  - X);
    double X3 = delta[d] - A3 * fabs(C3 * delta[d] - X);

    // Moyenne (nouvelle position)
    double Xnew = (X1 + X2 + X3) / 3.0;

    // Clamp dans les bornes
    Xnew = fmin(fmax(Xnew, LOWER_BOUND), UPPER_BOUND);

    new_positions[idx] = Xnew;

    // Sauvegarde RNG
    rngStates[idx] = localState;
}

// =======================================================
// CALCUL FITNESS (1 thread = 1 loup)
// =======================================================
__global__ void kernelComputeFitness(double *positions, double *fitness, int pop, int max_dim, int selected_obj_func)
{
    int wolf = blockIdx.x * blockDim.x + threadIdx.x;
    if (wolf >= pop) return;

    fitness[wolf] = device_fitness_function(
        &positions[wolf * max_dim], max_dim, selected_obj_func
    );
}

// =======================================================
// SÉLECTION ÉLITISTE
// =======================================================
/*
On garde la meilleure solution entre :
- ancienne position
- nouvelle position
*/
__global__ void kernelSelectBetter(
    double *oldPos,
    double *newPos,
    double *fitnessOld,
    double *fitnessNew,
    int pop,
    int max_dim
)
{
    int wolf = blockIdx.x * blockDim.x + threadIdx.x;
    if (wolf >= pop) return;

    if (fitnessNew[wolf] < fitnessOld[wolf]) {
        for (int d = 0; d < max_dim; d++) {
            oldPos[wolf * max_dim + d] =
                newPos[wolf * max_dim + d];
        }
        fitnessOld[wolf] = fitnessNew[wolf];
    }
}

// =======================================================
// INITIALISATION POPULATION
// =======================================================
__global__ void kernelInitPopulation(
    double *positions,
    curandState *rngStates,
    int total
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total)
    {
        curandState local = rngStates[idx];

        // Tirage uniforme dans l’espace de recherche
        double r = curand_uniform_double(&local);
        positions[idx] = LOWER_BOUND + (UPPER_BOUND - LOWER_BOUND) * r;

        rngStates[idx] = local;
    }
}

// =======================================================
// STRUCTURE BEST3
// =======================================================
/*
Stocke les 3 meilleurs individus
*/
struct Best3 {
    double val[3];
    int idx[3];
};

// =======================================================
// INSERTION DANS TOP 3
// Objectif :
// Maintenir dynamiquement les 3 meilleures valeurs (minimisation)
// dans la structure Best3 :

//     b.val[0] → meilleur (alpha)
//     b.val[1] → 2e meilleur (beta)
//     b.val[2] → 3e meilleur (delta)

// IMPORTANT :
// On maintient en permanence l'invariant suivant :
//     b.val[0] <= b.val[1] <= b.val[2]

// Cette fonction est appelée très souvent → doit être O(1)
// (pas de tri complet, uniquement insertion locale).
// =======================================================
__device__ void insertBest(Best3 &b, double val, int idx) {

    // =====================================================
    // CAS 1 : la nouvelle valeur est la meilleure (alpha)
    // =====================================================
    /*
    Si val est plus petite que le meilleur actuel,
    alors elle devient le nouveau alpha.

    On doit décaler :
        alpha → beta
        beta  → delta
    */
    if (val < b.val[0]) {

        // L'ancien beta devient delta (3e position)
        b.val[2] = b.val[1]; 
        b.idx[2] = b.idx[1];

        // L'ancien alpha devient beta (2e position)
        b.val[1] = b.val[0]; 
        b.idx[1] = b.idx[0];

        // Insertion du nouveau meilleur en alpha (1ère position)
        b.val[0] = val;      
        b.idx[0] = idx;
    }


    // =====================================================
    // CAS 2 : meilleur que beta mais pas alpha
    // =====================================================
    /*
    Ici :
    - val >= alpha → donc pas le meilleur
    - val < beta   → devient le 2e meilleur

    On décale :
        beta → delta
    */
    else if (val < b.val[1]) {

        // L'ancien beta devient delta
        b.val[2] = b.val[1]; 
        b.idx[2] = b.idx[1];

        // Insertion en position beta
        b.val[1] = val;      
        b.idx[1] = idx;
    }


    // =====================================================
    // CAS 3 : meilleur que delta uniquement
    // =====================================================
    /*
    Ici :
    - val >= alpha
    - val >= beta
    - val < delta

    Donc val devient simplement le 3e meilleur
    */
    else if (val < b.val[2]) {

        // Remplacement direct du delta
        b.val[2] = val;      
        b.idx[2] = idx;
    }
}

// =======================================================
// BEST3 PAR BLOC (réduction locale)
// Objectif :
//  Trouver les 3 meilleurs loups (fitness minimales)
//  POUR CHAQUE BLOC CUDA.

//  → Chaque bloc produit un Best3 local
//  → Ces résultats seront ensuite fusionnés globalement

//  Stratégie :
//  1. Chaque thread traite plusieurs loups (local)
//  2. Stockage dans shared memory
//  3. Réduction parallèle intra-bloc
// =======================================================
__global__ void kernelFindBest3(double *fitness, Best3 *blockBest, int pop) {

    // =====================================================
    // MÉMOIRE PARTAGÉE (rapide, accessible par le bloc)
    // =====================================================

    /*
    shared[] contient un Best3 par thread.

    Taille dynamique allouée au lancement du kernel :
        blockDim.x * sizeof(Best3)
    */
    extern __shared__ Best3 shared[];


    // =====================================================
    // IDENTIFIANT DU THREAD
    // =====================================================

    // ID local du thread dans le bloc
    int tid = threadIdx.x;

    // =====================================================
    // CHARGE DE TRAVAIL PAR THREAD
    // =====================================================

    /*
    Chaque thread va traiter plusieurs loups
    pour améliorer l’occupation GPU et réduire les accès mémoire.

    Ici : 5 loups par thread
    */
    int number_of_wolf = 5;


    // =====================================================
    // CALCUL DE L’INDEX DE DÉPART
    // =====================================================

    /*
    On découpe le tableau fitness en blocs.

    Chaque bloc traite :
        blockDim.x * number_of_wolf loups

    Chaque thread traite :
        number_of_wolf loups

    start = position de départ dans le tableau global
    */
    int start = blockIdx.x * blockDim.x * number_of_wolf
              + tid * number_of_wolf;


    // =====================================================
    // INITIALISATION DU TOP 3 LOCAL
    // =====================================================

    /*
    Chaque thread maintient ses propres 3 meilleurs loups
    (dans la variable local)
    */
    Best3 local;

    // Initialisation avec une valeur très grande (≈ +∞)
    local.val[0] = local.val[1] = local.val[2] = 1e20;


    // =====================================================
    // PARCOURS DES LOUPS ASSIGNÉS AU THREAD
    // =====================================================

    /*
    Chaque thread va traiter "number_of_wolf" loups
    et mettre à jour son top 3 local
    */
    for (int j = 0; j < number_of_wolf; j++) {

        // Index global du loup
        int i = start + j;

        // Vérification pour éviter dépassement mémoire
        if (i < pop)

            // Insertion dans le top 3 local
            insertBest(local, fitness[i], i);
    }


    // =====================================================
    // STOCKAGE EN MÉMOIRE PARTAGÉE
    // =====================================================

    /*
    Chaque thread écrit son Best3 dans shared memory

    → Permet une réduction rapide ensuite
    */
    shared[tid] = local;


    // Synchronisation obligatoire :
    // tous les threads doivent avoir écrit avant la réduction
    __syncthreads();


    // =====================================================
    // RÉDUCTION PARALLÈLE (intra-bloc)
    // =====================================================

    /*
    Objectif :
    Fusionner tous les Best3 des threads du bloc
    pour obtenir un seul Best3 par bloc.

    Méthode :
    réduction en arbre (tree reduction)
    */

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {

        /*
        À chaque étape :
        - moitié des threads travaillent
        - ils fusionnent avec leur "voisin"
        */

        if (tid < stride && tid + stride < blockDim.x) {

            /*
            Fusion des Best3 :
            On insère les 3 valeurs du voisin
            dans le Best3 courant
            */
            for (int k = 0; k < 3; k++) {

                insertBest(shared[tid],
                           shared[tid + stride].val[k],
                           shared[tid + stride].idx[k]);
            }
        }

        // Synchronisation après chaque étape de réduction
        __syncthreads();
    }


    // =====================================================
    // ÉCRITURE DU RÉSULTAT FINAL DU BLOC
    // =====================================================

    /*
    Après la réduction :
    → shared[0] contient le Best3 du bloc entier

    Un seul thread écrit le résultat global du bloc
    */
    if (tid == 0)

        blockBest[blockIdx.x] = shared[0];
}

// =======================================================
// RÉDUCTION GLOBALE
// Objectif :
// Fusionner les résultats provenant de tous les blocs
// (calculés dans kernelFindBest3) afin d’obtenir :

//     → les 3 meilleurs loups GLOBAUX

// Entrée :
//     input  = tableau de Best3 (1 par bloc)
//     n      = nombre de blocs

// Sortie :
//     output[0] = Best3 global (alpha, beta, delta)

// IMPORTANT :
// Ce kernel est lancé avec UN SEUL BLOC.
// → Toute la réduction se fait dans ce bloc.
// =======================================================
__global__ void kernelReduceBest3(Best3 *input, Best3 *output, int n) {

    // =====================================================
    // MÉMOIRE PARTAGÉE
    // =====================================================

    /*
    Tableau shared :
    - contient une copie des Best3
    - permet une réduction rapide (beaucoup plus rapide que global memory)
    */
    extern __shared__ Best3 shared[];


    // =====================================================
    // IDENTIFIANT DU THREAD
    // =====================================================

    // Index du thread dans le bloc
    int tid = threadIdx.x;


    // =====================================================
    // INITIALISATION LOCALE
    // =====================================================

    /*
    Chaque thread initialise une structure locale Best3.

    On initialise avec une valeur très grande (≈ +∞)
    pour garantir que toute vraie valeur sera meilleure.
    */
    Best3 local;
    local.val[0] = local.val[1] = local.val[2] = 1e20;


    // =====================================================
    // CHARGEMENT DES DONNÉES
    // =====================================================

    /*
    Chaque thread copie UN élément de input vers local.

    MAIS :
    → seulement si tid < n (sinon hors limites)
    */
    if (tid < n)
        local = input[tid];


    // =====================================================
    // COPIE EN MÉMOIRE PARTAGÉE
    // =====================================================

    /*
    Chaque thread écrit son Best3 dans shared memory.

    Cela permet ensuite une réduction parallèle rapide.
    */
    shared[tid] = local;


    // Synchronisation obligatoire :
    // tous les threads doivent avoir écrit avant la réduction
    __syncthreads();


    // =====================================================
    // RÉDUCTION PARALLÈLE (TREE REDUCTION)
    // =====================================================

    /*
    On réduit progressivement le tableau shared[] :

    À chaque étape :
    - on divise le nombre de threads actifs par 2
    - chaque thread fusionne avec un voisin

    Objectif :
    → shared[0] contiendra le meilleur Best3 global
    */

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {

        /*
        Condition pour participer à la réduction :
        - tid < stride → thread actif
        - tid + stride < n → évite accès hors tableau
        */
        if (tid < stride && tid + stride < n) {

            /*
            Fusion des Best3 :
            On prend les 3 valeurs du voisin (tid + stride)
            et on les insère dans le Best3 courant.

            → insertBest maintient automatiquement l’ordre
              alpha ≤ beta ≤ delta
            */
            for (int k = 0; k < 3; k++)
                insertBest(shared[tid],
                           shared[tid + stride].val[k],
                           shared[tid + stride].idx[k]);
        }

        // Synchronisation après chaque étape de réduction
        __syncthreads();
    }


    // =====================================================
    // ÉCRITURE DU RÉSULTAT FINAL
    // =====================================================

    /*
    À la fin :
    → shared[0] contient les 3 meilleurs globaux

    Un seul thread (tid == 0) écrit le résultat
    dans la mémoire globale.
    */
    if (tid == 0)
        output[0] = shared[0];
}

// =======================================================
// EXTRACTION ALPHA / BETA / DELTA
// =======================================================
__global__ void extractEliteFromBest3(
    double *positions,
    double *alpha,
    double *beta,
    double *delta,
    Best3 *best,
    int dim
) {
    // Index global du thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Vérification limites
    if (idx < dim) {

        // Récupération des indices des 3 meilleurs loups
        int alpha_idx = best[0].idx[0];
        int beta_idx  = best[0].idx[1];
        int delta_idx = best[0].idx[2];

        // Copie des dimensions correspondantes
        alpha[idx] = positions[alpha_idx * dim + idx];
        beta[idx]  = positions[beta_idx  * dim + idx];
        delta[idx] = positions[delta_idx * dim + idx];
    }
}

// =======================================================
// FONCTION PRINCIPALE CUDA GWO
// =======================================================
extern "C" double cuda_gwo(int pop, int dim, int max_epoch, int selected_obj_func, unsigned long seed)
{
    // Taille totale du tableau positions
    const int size = pop * dim;

    // Pointeurs GPU
    double *devPos, *devNewPos;
    double *devAlpha, *devBeta, *devDelta;
    double *devFitness, *devFitnessNew;
    curandState *devRngStates;

    // Allocation mémoire GPU
    cudaMalloc(&devPos,        sizeof(double) * size);
    cudaMalloc(&devNewPos,     sizeof(double) * size);
    cudaMalloc(&devAlpha,      sizeof(double) * dim);
    cudaMalloc(&devBeta,       sizeof(double) * dim);
    cudaMalloc(&devDelta,      sizeof(double) * dim);
    cudaMalloc(&devFitness,    sizeof(double) * pop);
    cudaMalloc(&devFitnessNew, sizeof(double) * pop);
    cudaMalloc(&devRngStates,  sizeof(curandState) * size);

    // Configuration GPU
    int threads = 256;
    int total = pop * dim;

    // Calcul du nombre de blocs
    int blocks = (total + threads - 1) / threads;
    int blocksRng    = (size + threads - 1) / threads;
    int blocksWolves = (pop  + threads - 1) / threads;

    // Initialisation RNG
    kernelInitRNG<<<blocksRng, threads>>>(devRngStates, seed, pop, dim);

    // Initialisation population
    kernelInitPopulation<<<blocks, threads>>>(devPos, devRngStates, total);

    // Synchronisation GPU
    cudaDeviceSynchronize();

    // Paramètre pour parallélisation Best3
    int K = 5;

    // Nombre de blocs pour Best3
    int blocksBest = (pop + threads*K - 1) / (threads*K);

    // Allocation Best3
    Best3 *devBlockBest, *devGlobalBest;
    cudaMalloc(&devBlockBest,  sizeof(Best3) * blocksBest);
    cudaMalloc(&devGlobalBest, sizeof(Best3));

    // Paramètre GWO
    double a_param;

    // Calcul fitness initiale
    kernelComputeFitness<<<blocksWolves, threads>>>(
        devPos, devFitness, pop, dim, selected_obj_func
    );

    // Boucle principale
    for (int iter = 0; iter < max_epoch; iter++)
    {
        // Recherche des 3 meilleurs
        kernelFindBest3<<<blocksBest, threads, threads * sizeof(Best3)>>>(devFitness, devBlockBest, pop);

        // Réduction globale
        kernelReduceBest3<<<1, threads, threads * sizeof(Best3)>>>(devBlockBest, devGlobalBest, blocksBest);

        // Extraction alpha, beta, delta
        int blocksDim = (dim + threads - 1) / threads;

        extractEliteFromBest3<<<blocksDim, threads>>>(
            devPos, devAlpha, devBeta, devDelta, devGlobalBest, dim
        );

        // Calcul du paramètre a (décroissance linéaire)
        a_param = 2.0 - 2.0 * ((double)iter / (double)max_epoch);

        // Mise à jour positions
        kernelUpdatePositionsGWO<<<blocks, threads>>>(
            devPos, devAlpha, devBeta, devDelta,
            devNewPos, a_param, devRngStates,
            pop, dim
        );

        // Calcul nouvelle fitness
        kernelComputeFitness<<<blocksWolves, threads>>>(
            devNewPos, devFitnessNew, pop, dim, selected_obj_func
        );

        // Sélection élitiste
        kernelSelectBetter<<<blocksWolves, threads>>>(
            devPos, devNewPos, devFitness, devFitnessNew, pop, dim
        );
    }

    // Calcul du meilleur final
    kernelFindBest3<<<blocksBest, threads, threads * sizeof(Best3)>>>(devFitness, devBlockBest, pop);

    kernelReduceBest3<<<1, threads, threads * sizeof(Best3)>>>(devBlockBest, devGlobalBest, blocksBest);

    // Copie résultat GPU → CPU
    Best3 hostBest;
    cudaMemcpy(&hostBest, devGlobalBest, sizeof(Best3), cudaMemcpyDeviceToHost);

    // Meilleure fitness
    double bestFitness = hostBest.val[0];

    // Libération mémoire GPU
    cudaFree(devPos);
    cudaFree(devNewPos);
    cudaFree(devAlpha);
    cudaFree(devBeta);
    cudaFree(devDelta);
    cudaFree(devFitness);
    cudaFree(devFitnessNew);
    cudaFree(devRngStates);
    cudaFree(devBlockBest);
    cudaFree(devGlobalBest);

    // Retour du résultat final
    return bestFitness;
}