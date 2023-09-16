#pragma once
#include <string>
#include "PolicyValue.h"
#include "mcts.h"
#include <deque>
#include <utility>
#include <string>
#include <random>
#include <algorithm>

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

class TrainPipeline {
private:
	const int n_playout = 400;
	const int buffer_size = 10000;
	const int play_batch_size = 1;
	const int epochs = 10;
	const int check_freq = 100;
	const int save_freq = 10;
	const int game_batch_num = 1500;
	float learning_rate = 0.002f;
	float lr_multiplier = 1.0f;
	float temp = 1.0f;
	float c_puct = 5.0f;

	std::deque<GameData> data_buffer;
	float kl_targ = 0.02f;
	int cnt = 0;
	int episode_len = 0;
	PolicyValueNet prev_policy;
	PolicyValueNet policy_value_net;
	MCTSPlayer mcts_player;

public:
	std::array<float, 5 * batchSize * largeSize>* state_batch;
	std::array<float, batchSize * (totSize + 1)>* mcts_probs;
	std::array<float, batchSize>* winner_batch;

	static std::pair<float, std::vector<GameData> > start_self_play(MCTSPlayer* player, bool is_shown = false, float temp = 0.1f);
	static float start_play(std::array<MCTSPlayer*, 2> player_list,
		bool is_shown = false, float temp = 0.1f);

	TrainPipeline(const std::string& init_model,
		const std::string& test_model, bool gpu = false, int cnt = 0);

	void get_equi_data(int s, int e);
	void collect_selfplay_data(int n_games = 1);
	void policy_update();
	float policy_evaluate(const std::string& process_type, bool is_shown=false, int n_games = 30);
	void run();
};