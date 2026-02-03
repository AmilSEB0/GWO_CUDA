# Grey Wolf Optimizer (GWO) - README

## Prérequis

Avant d'exécuter ce code, il est nécessaire de s'assurer que les éléments suivants sont installés sur le système :

1. **Compilateur C++ compatible avec la norme C++20 :** Ce projet a été développé en utilisant la norme C++20. Il est donc essentiel d'utiliser un compilateur compatible, tel que **g++ version 10** ou supérieur, ou un compilateur équivalent pour Windows (MinGW ou MSVC).

2. **CMake :** Il est nécessaire d'avoir CMake installé. CMake est un outil de gestion de la compilation qui permet de configurer et de compiler le code de manière portable.

3. **Terminal ou IDE :** Le code peut être exécuté soit via la ligne de commande, soit à l'aide d'un IDE tel que **CLion**, **Visual Studio Code** ou un autre IDE compatible avec CMake.

## Installation et Exécution

### Exécution sans CMake (Compilation manuelle avec g++)

Il est possible de compiler et exécuter le programme manuellement avec g++. Voici les étapes nécessaires :

1. Ouvrir un terminal et naviguer jusqu'au dossier contenant les fichiers du projet.

2. Compiler le code avec la commande g++ en utilisant la norme C++20 :

    ```bash
    g++ -std=c++20 -o gwo_optimizer main.cpp utils/Agent.cpp utils/Problem.cpp OriginalGWO.cpp
    ```

   Cette commande génère un fichier exécutable nommé **gwo_optimizer**.

3. Pour exécuter le programme, il suffit de saisir la commande suivante dans le terminal :

    ```bash
    ./gwo_optimizer
    ```

   Cette commande lancera le programme et effectuera l'optimisation.

### Exécution avec CMake

Il est aussi possible d'exécuter avec CMake, voici les étapes à suivre pour compiler et exécuter le programme :

1. Ouvrir un terminal et se rendre dans le dossier contenant les fichiers du projet.

2. Créer un répertoire de build pour la compilation :

    ```bash
    mkdir build
    cd build
    ```

3. Utiliser **CMake** pour configurer le projet et générer les fichiers nécessaires à la compilation :

    ```bash
    cmake ..
    ```

   Cela génère les fichiers de construction pour **g++** (ou un autre compilateur compatible) à partir du fichier **CMakeLists.txt**.

4. Compiler le projet avec **make** :

    ```bash
    make
    ```

   Cette étape crée l'exécutable **GWO_C**.

5. Une fois la compilation terminée, il est possible d'exécuter le programme en utilisant la commande suivante :

    ```bash
    ./GWO_C
    ```

   Cela lancera le programme et effectuera l'optimisation.

### Exécution avec un IDE (par exemple, CLion)

Si un IDE comme CLion est utilisé, voici la procédure pour exécuter le programme avec CMake :

1. Ouvrir le projet dans CLion :
    * Lancer CLion.
    * Sélectionner **Open** et choisir le dossier contenant le code source (le dossier où se trouve le fichier **CMakeLists.txt**).

2. Configurer CMake dans CLion :
    * CLion détectera automatiquement le fichier **CMakeLists.txt** et proposera de configurer le projet.
    * Il est important de s'assurer que CMake et un compilateur compatible C++20 (comme **g++ version 10** ou supérieur) sont correctement configurés dans les paramètres de CLion.

3. Compiler et exécuter dans CLion :
    * Une fois la configuration terminée, il est possible de cliquer sur le bouton **Run** dans CLion pour compiler et exécuter le programme.
    * **CLion** s'occupera de la gestion de la compilation via CMake et lancera l'exécution automatiquement.


Auteur : **SEBO Amil**  
Date : **20 février 2025**
