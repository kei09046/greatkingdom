#include "memory.h"
#include <array>
using namespace std;


GameData::GameData(array<float, 5 * largeSize>& _state,
	array<float, totSize + 1>& _mcts_probs, float _winner) :
	state(_state), mcts_probs(_mcts_probs), winner(_winner) {}

GameData::GameData(array<float, 5 * largeSize>&& _state,
	array<float, totSize + 1>&& _mcts_probs, float _winner) :
	state(_state), mcts_probs(_mcts_probs), winner(_winner) {}