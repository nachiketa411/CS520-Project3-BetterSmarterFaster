# After these number of moves, if nothing happens between the 3 idiots, then increment FAILURE_RATE_2
NO_OF_STEPS_1 = 200
NO_OF_STEPS_2 = 300
NO_OF_STEPS_3 = 500
NO_OF_STEPS_4 = 5000

NO_OF_GRAPHS = 100
NO_OF_RUNS = 30

NO_OF_NODES = 50

NO_OF_NEXT_STEP_PREDICTIONS_FOR_AGENT_2 = 1
PROB_OF_DISTRACTED_PREDATOR = 0.4

ENVIRONMENT_PATH = "Environment.json"
GRAPH_DIST_PATH = "Node Distances.json"
UTILITIES_PATH = "Utilities.npy"
WEIGHTS_PATH = "Weights.npy"

BETA = 1
REWARD = 1

NO_OF_TIMES_I_WANT_UTILITIES_TO_BE_CONSISTENT = 2

# -------------------------------------------------------
# Neural Network Properties:-
# -------------------------------------------------------
ALPHA = 10 ** -3
NO_OF_HIDDEN_LAYERS = 2
# the last value corresponds to the original number of outputs i.e. 1
NO_OF_HIDDEN_UNITS = [5, 3, 1]
INFINITY = 60.
INFINITE_THRESHOLD = 150.

BATCH_SIZE = 125000
BETA_1 = 0.9
BETA_2 = 0.99
EPSILON = 10 ** -9


