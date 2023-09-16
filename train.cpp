#include "PolicyValue.h"
#include "mcts.h"
#include "train.h"
using namespace std;


GameData::GameData(array<float, 5 * largeSize>& _state,
	array<float, totSize + 1>& _mcts_probs, float _winner): 
	state(_state), mcts_probs(_mcts_probs), winner(_winner) {}

GameData::GameData(array<float, 5 * largeSize>&& _state,
	array<float, totSize + 1>&& _mcts_probs, float _winner):
	state(_state), mcts_probs(_mcts_probs), winner(_winner) {}


pair<float, vector<GameData> > TrainPipeline::start_self_play(MCTSPlayer* player, bool is_shown, float temp){
	GameManager game_manager = GameManager();
	std::vector<GameData> datas;
	int result;
	bool b;
	std::pair<int, std::array<float, totSize + 1>> move_prob;

	while (true) {
		player->get_action(game_manager, move_prob, is_shown, temp);
		/*cout << "get_action" << endl;*/
		//cout << "current turn : " << game_manager.get_turn() << endl;
		datas.emplace_back(game_manager.current_state(), move_prob.second, 0.0f);
		//int x = 0;
		//for (int i = 0; i < 5; ++i) {
		//	for (int j = 0; j < largeSize; ++j)
		//		cout << datas.back().state[x++] << " ";
		//	cout << endl;
		//}

		//cout << move_prob.first / boardSize << ", " << move_prob.first % boardSize << endl;
		result = game_manager.make_move(move_prob.first, true);

		if (result) {
			// result : 흑이 이기면 1 백이 이기면 -1 
			if (result == -2)
				result = game_manager.end_game().first;
			else
				result *= game_manager.get_turn();

			b = result > 0;
			for (GameData& gd : datas) {
				if (b)
					gd.winner = 1.0f;
				else
					gd.winner = -1.0f;
				b = !b;
			}

			player->reset_player();
			for (auto m : game_manager.get_seqence())
				cout << m.first << "," << m.second << " ";
			cout << endl;
			return { result, datas };
		}

		game_manager.switch_turn();
	}
}

float TrainPipeline::start_play(array<MCTSPlayer*, 2> player_list, bool is_shown, float temp) {
	GameManager game_manager = GameManager();
	int t[3] = { -1, 0, 1 };
	int idx = 0;
	int move, res, diff;

	while (true) {
		player_list[idx]->get_action(game_manager, move, is_shown, temp);
		res = game_manager.make_move(move, true);
		game_manager.switch_turn();

		if (!res) {
			idx = 1 - idx;
			continue;
		}

		if (is_shown) {
			auto seq = game_manager.get_seqence();
			for (auto moves : seq)
				cout << moves.first << moves.second << " ";
		}

		if (res == -2) {
			std::tie(res, diff) = game_manager.end_game();
			switch (res) {
			case 0:
				cout << "draw" << endl;
				return 0.5f;
			case -1:
				cout << "second player wins" << endl;
				return 0.0f;
			case 1:
				cout << "first player wins" << endl;
				return 1.0f;
			}
		}

		if ((res == 1 && !idx) || (res == -1 && idx)) {
			cout << "first player wins" << endl;
			return 1.0f;
		}
		else {
			cout << "second player wins" << endl;
			return 0.0f;
		}
	}
}

TrainPipeline::TrainPipeline(const std::string& init_model,
	const std::string& test_model, bool gpu, int cnt): policy_value_net(init_model, gpu),
prev_policy(test_model, gpu), mcts_player(&policy_value_net, c_puct, n_playout, true), cnt(cnt){
	state_batch = new array<float, 5 * batchSize * largeSize>();
	mcts_probs = new array<float, batchSize* (totSize + 1)>();
	winner_batch = new array<float, batchSize>();
}

void TrainPipeline::get_equi_data(int s, int e)
{
	std::array<float, 5 * largeSize> temp_state;
	std::array<float, totSize + 1> temp_mcts_probs;
	int cnt, dnt;

	for (int i = s; i < e; ++i) {
		const GameData& gd = data_buffer[i];


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + l * (boardSize + 2) + k];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[l * boardSize + k];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + (k + 1) * (boardSize + 2) - (l + 1)];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[(k + 1) * boardSize - (l + 1)];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + (l + 1) * (boardSize + 2) - (k + 1)];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[(l + 1) * boardSize - (k + 1)];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + (boardSize + 1 - k) * (boardSize + 2) + l];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[(boardSize - 1 - k) * boardSize + l];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + (boardSize + 1 - l) * (boardSize + 2) + k];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[(boardSize - 1 - l) * boardSize + k];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + (boardSize + 2 - k) * (boardSize + 2) - (l + 1)];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[(boardSize - k) * boardSize - (l + 1)];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);


		cnt = 0;
		dnt = 0;
		for (int j = 0; j < 5; ++j)
			for (int k = 0; k < boardSize + 2; ++k)
				for (int l = 0; l < boardSize + 2; ++l)
					temp_state[cnt++] = gd.state[j * largeSize + (boardSize + 2 - l) * (boardSize + 2) - (k + 1)];

		for (int k = 0; k < boardSize; ++k)
			for (int l = 0; l < boardSize; ++l)
				temp_mcts_probs[dnt++] = gd.mcts_probs[(boardSize - l) * boardSize - (k + 1)];

		temp_mcts_probs[totSize] = gd.mcts_probs[totSize];
		data_buffer.emplace_back(std::move(temp_state), std::move(temp_mcts_probs), gd.winner);
	}

	int over = data_buffer.size() - buffer_size;
	// 개선 여지 : experience replay 과정에서 선택적으로 제거하기
	if(over > 0)
		data_buffer.erase(data_buffer.begin(), data_buffer.begin() + over);
	return;
}

void TrainPipeline::collect_selfplay_data(int n_games)
{
	pair<float, vector<GameData>> res;
	for (int i = 0; i < n_games; ++i) {
		res = TrainPipeline::start_self_play(&mcts_player, true, temp);
		cout << "episode length : " << res.second.size() << " winner : " << res.first << endl;

		for (GameData gd : res.second) {
			data_buffer.push_back(move(gd));
			//cout << gd.winner << endl;
		}
		
		get_equi_data(data_buffer.size() - res.second.size(), data_buffer.size());
	}
	return;
}

void TrainPipeline::policy_update()
{
	/*cout << " shuffle " << endl;*/
	random_device rd;
	mt19937 mt(rd());
	//int cnt = 0;
	vector<int> sh(data_buffer.size());
	for (int i = 0; i < sh.size(); ++i)
		sh[i] = i;

	//Fischer-Yates shuffle
	/*cout << "shuffle start " << endl;*/
	/*int currentIndexCounter = data_buffer.size();
	for (auto iter = shuffle.rbegin(); iter != shuffle.rend(); ++iter, --currentIndexCounter) {
		if (++cnt > batchSize)
			break;

		uniform_int_distribution<> dis(0, currentIndexCounter);
		const int randomIndex = dis(mt);

		if (*iter != shuffle.at(randomIndex))
			swap(shuffle[randomIndex], *iter);
	}*/

	shuffle(begin(sh), end(sh), mt);
	/*cout << "shuffle end" << endl;*/
	int x = 0, y = 0;
	for (int i = 0; i < batchSize; ++i) {
		for (int j = 0; j < 5 * largeSize; ++j)
			(*state_batch)[x++] = data_buffer[sh[i]].state[j];
		for (int j = 0; j < totSize + 1; ++j)
			(*mcts_probs)[y++] = data_buffer[sh[i]].mcts_probs[j];

		(*winner_batch)[i] = data_buffer[sh[i]].winner;
		//cout << winner_batch[i] << endl;
	}

	auto old_probs_value = new pair<array<float, batchSize* (totSize + 1)>, array<float, batchSize>>(policy_value_net.policy_value(state_batch));
	auto new_probs_value = new pair<array<float, batchSize* (totSize + 1)>, array<float, batchSize>>();

	float ov, nv;
	float kl = 0.0f;
	for (int i = 0; i < epochs; ++i) {
		/*cout << " train step " << endl;*/
		kl = 0.0f;
		policy_value_net.train_step(*state_batch, *mcts_probs, *winner_batch, learning_rate * lr_multiplier);
		*new_probs_value = policy_value_net.policy_value(state_batch);
		for (int j = 0; j < batchSize * (totSize + 1); ++j) {
			ov = old_probs_value->first[j];
			nv = new_probs_value->first[j];
			kl += ov * (log(ov + 0.0001f) - log(nv + 0.0001f));
		}
		kl /= batchSize;
		if (kl > kl_targ * 4)
			break;
	}

	if (kl > kl_targ * 2 && lr_multiplier > 0.1f)
		lr_multiplier /= 1.5f;
	else if (kl < kl_targ / 2 && lr_multiplier < 10.0f)
		lr_multiplier *= 1.5f;

	delete old_probs_value;
	delete new_probs_value;
	return;
}

// 추가 개선가능점 : process type에 따라 몬테카를로 탐색에서 멀티스레드 적용
float TrainPipeline::policy_evaluate(const std::string& process_type, bool is_shown, int n_games)
{
	MCTSPlayer* current_player = new MCTSPlayer(&policy_value_net, c_puct, n_playout, false);
	MCTSPlayer* past_player = new MCTSPlayer(&prev_policy, c_puct, n_playout, false);
	float win_cnt = 0.0f;

	//cout << n_games << endl;
	for (int i = 0; i < n_games; ++i) {
		if (!(i % 2))
			win_cnt += TrainPipeline::start_play({ current_player, past_player }, is_shown, temp);
		else
			win_cnt += 1.0f - TrainPipeline::start_play({ past_player, current_player }, is_shown, temp);
	}

	delete current_player;
	delete past_player;
	return win_cnt / static_cast<float>(n_games);
}

void TrainPipeline::run()
{
	string model_file;

	for (int i = 0; i < game_batch_num; ++i) {
		collect_selfplay_data(play_batch_size);
		if (data_buffer.size() > batchSize)
			policy_update();

		if (!((i + 1 + cnt) % save_freq)) {
			model_file = "model3b";
			model_file += to_string(i + 1 + cnt);
			policy_value_net.save_model(model_file + string(".pt"));
			cout << "saved" << endl;
		}

		if (!((i + 1 + cnt) % check_freq)) {
			cout << "current self-play-batch: " << i + cnt << endl;
			float win_ratio = policy_evaluate("single", true);
			cout << win_ratio << endl;

			if (win_ratio > 0.55f) {
				model_file += string("best.pt");
				policy_value_net.save_model(model_file);
				prev_policy.load_model(model_file);
			}
			else {
				prev_policy.save_model("model.pt");
				policy_value_net.load_model("model.pt");
			}
		}
	}
}

