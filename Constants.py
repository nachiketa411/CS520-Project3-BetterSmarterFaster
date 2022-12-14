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
U_PARTIAL_UTILITIES = "UPartial Utilities.npy"
V_MODEL_UTILITIES_PATH = "VModel Utility.npy"
V_MODEL_SIX_LAYER_UTILITIES_PATH = "V Model 6 layer Utility.npy"
V_PARTIAL_6_LAYER_UTILITIES_PATH = "V Partial 6 layer Utility.npy"
TWO_LAYER_WEIGHTS_PATH = "WeightsWithTwoLayers.npy"
ONE_LAYER_WEIGHTS_PATH = "WeightsWithOneLayer.npy"
SIX_LAYER_WEIGHTS_PATH = "WeightsWithSixLayers.npy"
ONE_LAYER_ONE_INPUT_WEIGHTS_PATH = "WeightsWithOneLayerOneInput.npy"
V_PARTIAL_6_LAYER_WEIGHTS_PATH = "WeightsWithSixLayerForPartialCase.npy"
V_PARTIAL_6_LAYER_WITH_54_INPUTS_WEIGHTS_PATH = "Weights With 54 Inputs SixLayer For PartialCase.npy"
V_PARTIAL_2_LAYER_WITH_54_INPUTS_WEIGHTS_PATH = "Weights With 54 Inputs TwoLayer For PartialCase.npy"

BETA = 1
REWARD = 1

NO_OF_TIMES_I_WANT_UTILITIES_TO_BE_CONSISTENT = 2

# -------------------------------------------------------
# Neural Network Properties:-
# -------------------------------------------------------
ALPHA = 10 ** -4
# NO_OF_HIDDEN_LAYERS = 2
# NO_OF_HIDDEN_LAYERS = 1
# NO_OF_HIDDEN_LAYERS = 6
NO_OF_HIDDEN_LAYERS = 2
# the last value corresponds to the original number of outputs i.e. 1
# NO_OF_HIDDEN_UNITS = [5, 3, 1]
# NO_OF_HIDDEN_UNITS = [5, 1]
# NO_OF_HIDDEN_UNITS = [5, 8, 6, 4, 10, 5, 1]
NO_OF_HIDDEN_UNITS = [60, 60, 1]
INFINITY = 60.

# BATCH_SIZE = 2439
# 4741 for V-Partial since total no. of inputs = 194381
BATCH_SIZE = 4741
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 10 ** -20


