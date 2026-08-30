# Grey Wolf Optimizer (GWO) - Implémentation CUDA

## Description

Ce projet implémente l’algorithme d’optimisation **Grey Wolf Optimizer (GWO)** en utilisant CUDA afin d’exploiter le parallélisme GPU.

L’objectif est d’accélérer les calculs par rapport à une version séquentielle CPU, notamment pour :
- l’évaluation des fonctions de fitness
- la mise à jour des positions
- la sélection des meilleurs individus

Plusieurs fonctions de test sont utilisées :
- Rastrigin
- Rosenbrock
- Ackley

L’objectif est également d’évaluer les performances de l’implémentation GPU en termes de qualité des solutions et de temps d’exécution, sur différentes dimensions et tailles de population.

## Prérequis

Avant d'exécuter ce code, assurez-vous d'avoir installé les éléments suivants sur votre système :

1. **CUDA Toolkit et pilotes NVIDIA :**  
   Ce projet utilise CUDA pour l’accélération GPU. Installez le **CUDA Toolkit 12.4** et le **pilote NVIDIA 560.94** ou supérieur. Le compilateur utilisé est **nvcc 12.4**.

2. **Compilateur C++ compatible avec C++17 :**  
   Le code nécessite un compilateur C++ compatible avec la norme C++17 (MSVC via Visual Studio 2022 sur Windows).  

3. **CMake :**  
   CMake 3.18 ou supérieur est nécessaire pour la compilation et la configuration du projet.  

4. **Terminal ou IDE :**  
   Vous pouvez utiliser la **ligne de commande Windows** (X64 Native Tools Command Prompt for VS 2022) ou un IDE compatible avec CMake, comme **CLion** ou **Visual Studio**.

---

## Configuration matérielle utilisée

Les expériences ont été réalisées sur une machine de bureau avec :  

- **CPU :** Intel Core i5-2500, 3,30 GHz, 4 cœurs physiques  
- **RAM :** 16 Go  
- **GPU :** NVIDIA GeForce GTX 960, 4 Go de mémoire globale  
- **OS :** Windows 10 Professionnel (version 10.0.19045)  
- **CUDA :** Toolkit 12.4, pilote 560.94  

---

## Compilation et Exécution

### 1. Compilation avec nvcc (ligne de commande Windows)

1. Ouvrir le **X64 Native Tools Command Prompt for VS 2022** :  
   - Cliquer sur le menu Windows, taper « X64 Native Tools Command Prompt for VS 2022 » et ouvrir.  

2. Se rendre dans le dossier du projet :  

    ```cmd
    cd /d "C:\Users\ad\Desktop\calcul massive parallèle\GWO"
    ```

3. Compiler les fichiers CUDA et C++ avec **nvcc** :  

    ```cmd
    nvcc main.cpp kernel.cu -o gwo.exe
    ```

    Cette commande génère l’exécutable **gwo.exe**.  

4. Exécuter le programme :  

    ```cmd
    gwo.exe
    ```

---

### 2. Compilation avec CMake

Le projet peut également être compilé avec CMake, qui gère automatiquement CUDA et C++.

1. Ouvrir un terminal (ou PowerShell) et se placer dans le dossier du projet :  

    ```cmd
    cd /d "C:\Users\ad\Desktop\calcul massive parallèle\GWO"
    ```

2. Créer un dossier de compilation :  

    ```cmd
    mkdir build
    cd build
    ```

3. Configurer le projet avec CMake :  

    ```cmd
    cmake ..
    ```

4. Compiler avec `cmake --build` :  

    ```cmd
    cmake --build . --config Release
    ```

    - L’exécutable généré sera **mon_exec.exe**.  

5. Lancer le programme :  

    ```cmd
    mon_exec.exe
    ```

---

### 3. Exécution avec un IDE (Visual Studio ou CLion)

Si vous utilisez un IDE compatible avec CMake et CUDA :  

1. **Ouvrir le projet dans l’IDE :**  
   - Choisir le dossier contenant `CMakeLists.txt`.  

2. **Configurer CMake et CUDA :**  
   - L’IDE détectera automatiquement CUDA et le compilateur MSVC.  
   - Vérifier que CMake utilise bien le standard C++17 et CUDA 17.  

3. **Compiler et exécuter :**  
   - Cliquer sur **Run** ou **Build & Run** selon l’IDE.  
   - L’exécutable sera généré et lancé automatiquement.

---

## Résultats

Le programme génère deux fichiers CSV :

- `execution_results.csv` :
  Contient les résultats détaillés de chaque exécution (fitness finale et temps d’exécution).

- `mean_std_fitness.csv` :
  Contient les statistiques (moyenne et écart-type des fitness et des temps).

## Accélération GPU (CUDA)

Cette version utilise CUDA pour :

- paralléliser le calcul des fitness
- accélérer la mise à jour des positions des loups
- effectuer la sélection des meilleurs individus en parallèle

Cela permet de réduire significativement le temps d’exécution par rapport à une version CPU séquentielle.

---

Auteur : **SEBO Amil**  
Date : **16 avril 2026** 