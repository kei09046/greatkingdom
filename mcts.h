#pragma once

#include <utility>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <functional>
#include "PolicyValue.h"

class MCTS_node {
private:
	bool leaf = true;

public:
	std::array<MCTS_node*, totSize + 1> children;
	MCTS_node* parent;
	int _n_visits;
	float _Q, _U, _P;

	MCTS_node(MCTS_node* parent, float prior_p);
	//~MCTS_node();
	void expand(const std::array<float, totSize + 1>& probs);
	float get_value(float c_puct);
	void update(float leaf_value);
	void update_recursive(float leaf_value);
	int select(float c_puct);
	bool is_leaf() const;
	bool is_root() const;
};

class MCTS {
private:
	PolicyValueNet* pv;
	const float _c_puct;
	const int _n_playout;
	MCTS_node* root;

public:
	MCTS(PolicyValueNet* net, float c_puct = 5.0f, int n_playout = 10000);
	~MCTS();
	void delete_tree(MCTS_node* base);
	void _playout(GameManager game_manager);
	std::array<float, totSize + 1> get_move_probs(const GameManager& game_manager, float temp=0.1, bool is_shown=false);
	void update_with_move(int last_move);
};

class MCTSPlayer {
private:
	const bool _is_selfplay;
	bool player;
	MCTS mcts;
	float get_random();

public:
	MCTSPlayer(PolicyValueNet* net, int c_puct=5, int n_playout=2000, bool is_selfplay=false);
	void set_player_ind(bool p);
	void reset_player();
	void get_action(const GameManager& game_manager, int& r, bool shown = false, float temp = 0.1f);
	void get_action(const GameManager& game_manager, std::pair<int, std::array<float, totSize + 1> >& r, bool shown = false, float temp = 0.1f);
};