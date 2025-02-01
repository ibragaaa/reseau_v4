#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =====================================================
// Définitions et constantes
// =====================================================

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

// Transformation : nos points sont en « coordonnées mathématiques » 
// On les place autour du centre de la fenêtre avec un facteur d'échelle
#define SCALE 20.0
#define OFFSET_X (WINDOW_WIDTH / 2)
#define OFFSET_Y (WINDOW_HEIGHT / 2)

// Configuration du réseau de neurones
#define NUM_HIDDEN 5    // Essayez d’augmenter ce nombre (par ex. 10 ou 20) pour le problème des spirales
#define NUM_OUTPUT 2
// Pour la phase d'apprentissage (nombre d'itérations par frame)
#define TRAINING_ITERATIONS_PER_FRAME 100

// Pour la génération des spirales
#define DT 0.2
#define T_MAX 18.0  // Cela génère environ (T_MAX/DT + 1) points par spirale

// =====================================================
// Structures
// =====================================================

// Un point de notre jeu de données
typedef struct {
    double x;
    double y;
    int label; // 0 : spirale bleue, 1 : spirale rouge
} Point;

// Un neurone avec biais
typedef struct {
    int n_inputs;              // nombre d'entrées (sans compter le biais)
    double weights[10];        // tableau de poids (taille fixe suffisante pour ce projet)
    double bias;               // biais du neurone
    double output;             // sortie calculée
    double delta;              // delta utilisé pour la rétropropagation
} Neuron;

// =====================================================
// Fonctions utilitaires
// =====================================================

// Retourne un nombre aléatoire dans [-1, 1]
double random_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Fonction d'activation : tanh
double activation(double x) {
    return tanh(x);
}

// Dérivée de tanh en fonction de la sortie déjà calculée (car tanh'(x)=1-tanh²(x))
double activation_derivative_from_output(double output) {
    return 1.0 - output * output;
}

// Calcule la sortie d'un neurone à partir d'un vecteur d'entrées (ajoute le biais)
double neuron_compute(Neuron *neuron, double inputs[]) {
    double sum = neuron->bias;
    for (int i = 0; i < neuron->n_inputs; i++) {
        sum += neuron->weights[i] * inputs[i];
    }
    return activation(sum);
}

// =====================================================
// Propagation avant (feedforward)
// =====================================================

void feedforward(double input[2], Neuron hidden[], Neuron output[]) {
    double hidden_outputs[NUM_HIDDEN];
    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden[i].output = neuron_compute(&hidden[i], input);
        hidden_outputs[i] = hidden[i].output;
    }
    for (int j = 0; j < NUM_OUTPUT; j++) {
        output[j].output = neuron_compute(&output[j], hidden_outputs);
    }
}

// =====================================================
// Phase d'apprentissage : rétropropagation avec biais
// =====================================================

void train_network(Point training_set[], int training_set_size, Neuron hidden[], Neuron output[], double learning_rate) {
    // Sélection d'un exemple aléatoire
    int idx = rand() % training_set_size;
    double input[2] = { training_set[idx].x, training_set[idx].y };

    // Propagation avant : calcul des sorties
    double hidden_outputs[NUM_HIDDEN];
    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden[i].output = neuron_compute(&hidden[i], input);
        hidden_outputs[i] = hidden[i].output;
    }
    for (int j = 0; j < NUM_OUTPUT; j++) {
        output[j].output = neuron_compute(&output[j], hidden_outputs);
    }

    // Définition des cibles :
    // Pour la spirale bleue (label 0) : cible = [1, -1]
    // Pour la spirale rouge (label 1) : cible = [-1, 1]
    double target[NUM_OUTPUT];
    if (training_set[idx].label == 0) {
        target[0] = 1.0;
        target[1] = -1.0;
    } else {
        target[0] = -1.0;
        target[1] = 1.0;
    }

    // Calcul des deltas pour la couche de sortie
    for (int j = 0; j < NUM_OUTPUT; j++) {
        double out = output[j].output;
        output[j].delta = activation_derivative_from_output(out) * (target[j] - out);
    }

    // Calcul des deltas pour la couche cachée
    for (int i = 0; i < NUM_HIDDEN; i++) {
        double out = hidden[i].output;
        double sum = 0.0;
        for (int j = 0; j < NUM_OUTPUT; j++) {
            sum += output[j].weights[i] * output[j].delta;
        }
        hidden[i].delta = activation_derivative_from_output(out) * sum;
    }

    // Mise à jour des poids et biais de la couche de sortie
    for (int j = 0; j < NUM_OUTPUT; j++) {
        for (int i = 0; i < output[j].n_inputs; i++) {
            output[j].weights[i] += learning_rate * output[j].delta * hidden_outputs[i];
        }
        output[j].bias += learning_rate * output[j].delta;
    }

    // Mise à jour des poids et biais de la couche cachée
    for (int i = 0; i < NUM_HIDDEN; i++) {
        for (int j = 0; j < hidden[i].n_inputs; j++) {
            hidden[i].weights[j] += learning_rate * hidden[i].delta * input[j];
        }
        hidden[i].bias += learning_rate * hidden[i].delta;
    }
}

// =====================================================
// Fonctions de dessin
// =====================================================

void draw_filled_circle(SDL_Renderer *renderer, int x0, int y0, int radius) {
    for (int w = -radius; w <= radius; w++) {
        for (int h = -radius; h <= radius; h++) {
            if (w * w + h * h <= radius * radius) {
                SDL_RenderDrawPoint(renderer, x0 + w, y0 + h);
            }
        }
    }
}

// =====================================================
// Fonction principale
// =====================================================

int main(int argc, char* argv[]) {
    // Initialisation de SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Erreur SDL : %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Réseau de Neurones - Spirales",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          WINDOW_WIDTH, WINDOW_HEIGHT,
                                          SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Erreur création fenêtre : %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Erreur création renderer : %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    srand((unsigned int)time(NULL));

    // =====================================================
    // Création du jeu de données : deux spirales
    // =====================================================
    int num_points = (int)(T_MAX / DT) + 1;
    int training_set_size = num_points * 2; // une spirale bleue et une spirale rouge
    Point *training_set = malloc(training_set_size * sizeof(Point));
    if (!training_set) {
        fprintf(stderr, "Erreur d'allocation mémoire pour le jeu de données\n");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    int index = 0;
    for (double t = 0; t <= T_MAX; t += DT) {
        // Spirale bleue : x = t*cos(t), y = t*sin(t)
        training_set[index].x = t * cos(t);
        training_set[index].y = t * sin(t);
        training_set[index].label = 0; // bleue
        index++;

        // Spirale rouge : x = -t*cos(t), y = -t*sin(t)
        training_set[index].x = -t * cos(t);
        training_set[index].y = -t * sin(t);
        training_set[index].label = 1; // rouge
        index++;
    }

    // =====================================================
    // Initialisation du réseau de neurones
    // =====================================================

    // Couche cachée : chaque neurone reçoit 2 entrées (x et y)
    Neuron hidden[NUM_HIDDEN];
    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden[i].n_inputs = 2;
        for (int j = 0; j < hidden[i].n_inputs; j++) {
            hidden[i].weights[j] = random_weight();
        }
        hidden[i].bias = random_weight();
    }

    // Couche de sortie : chaque neurone reçoit NUM_HIDDEN entrées (les sorties de la couche cachée)
    Neuron output_layer[NUM_OUTPUT];
    for (int i = 0; i < NUM_OUTPUT; i++) {
        output_layer[i].n_inputs = NUM_HIDDEN;
        for (int j = 0; j < output_layer[i].n_inputs; j++) {
            output_layer[i].weights[j] = random_weight();
        }
        output_layer[i].bias = random_weight();
    }

    double learning_rate = 0.01;
    int quit = 0;
    SDL_Event event;

    // Boucle principale
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = 1;
        }

        // Effectuer quelques itérations d'apprentissage par frame
        for (int i = 0; i < TRAINING_ITERATIONS_PER_FRAME; i++) {
            train_network(training_set, training_set_size, hidden, output_layer, learning_rate);
        }

        // Effacer l'écran (fond noir)
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // =====================================================
        // Affichage du « decision surface »
        // Pour chaque cellule d'une grille (ici 4x4 pixels), calculer la sortie du réseau.
        // =====================================================
        for (int screen_y = 0; screen_y < WINDOW_HEIGHT; screen_y += 4) {
            for (int screen_x = 0; screen_x < WINDOW_WIDTH; screen_x += 4) {
                double x = (screen_x - OFFSET_X) / SCALE;
                double y = (OFFSET_Y - screen_y) / SCALE;  // inversion de Y
                double input[2] = { x, y };

                double hidden_outputs[NUM_HIDDEN];
                for (int i = 0; i < NUM_HIDDEN; i++) {
                    hidden_outputs[i] = neuron_compute(&hidden[i], input);
                }
                double outputs[NUM_OUTPUT];
                for (int j = 0; j < NUM_OUTPUT; j++) {
                    outputs[j] = neuron_compute(&output_layer[j], hidden_outputs);
                }

                // Conversion des sorties en intensités de couleur
                double p_blue = (outputs[0] + 1.0) / 2.0;
                double p_red  = (outputs[1] + 1.0) / 2.0;
                Uint8 red   = (Uint8)(p_red * 255);
                Uint8 blue  = (Uint8)(p_blue * 255);
                SDL_SetRenderDrawColor(renderer, red, 0, blue, 255);
                SDL_Rect rect = { screen_x, screen_y, 4, 4 };
                SDL_RenderFillRect(renderer, &rect);
            }
        }

        // =====================================================
        // Dessin des points d'apprentissage (les spirales)
        // =====================================================
        for (int i = 0; i < training_set_size; i++) {
            int px = OFFSET_X + (int)(training_set[i].x * SCALE);
            int py = OFFSET_Y - (int)(training_set[i].y * SCALE);
            if (training_set[i].label == 0)
                SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // bleue
            else
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // rouge
            draw_filled_circle(renderer, px, py, 3);
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16); // ~60 FPS
    }

    // Libération des ressources
    free(training_set);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
