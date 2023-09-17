#pragma once
// source : https://github.com/rlcode/per/tree/master

#include <array>
#include "mcts.h"
#include "GameManager.h"

class GameData {
public:
	std::array<float, 5 * largeSize> state;
	std::array<float, totSize + 1> mcts_probs;
	float winner;

	GameData(std::array<float, 5 * largeSize>& _state,
		std::array<float, totSize + 1>& _mcts_probs, float _winner);
	GameData(std::array<float, 5 * largeSize>&& _state,
		std::array<float, totSize + 1>&& _mcts_probs, float _winner);
};

//class SumTree {
//public:
//	const static int capacity = 1 << 14;
//	SumTree(int capacity);
//	void propagate(int idx, float delta);
//	void update(int idx, float prior);
//	void emplace_back(float prior, std::array<float, 5 * largeSize>& _state,
//		std::array<float, totSize + 1>& _mcts_probs, float _winner);
//	void emplace_back(float prior, std::array<float, 5 * largeSize>&& _state,
//		std::array<float, totSize + 1>&& _mcts_probs, float _winner);
//	int find(int idx, float s) const;
//	float total() const;
//
//private:
//	float tree[capacity * 2 - 1];
//	int loc = 0;
//	GameData datas[capacity];
//};
//
//class Memory {
//
//};