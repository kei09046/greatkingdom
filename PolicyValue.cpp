#include "PolicyValue.h"
#include "GameManager.h"
#include <cmath>
#include <iostream>
using namespace std;


NetImpl::NetImpl(bool use_gpu = false): cv1(torch::nn::Conv2dOptions(5, 32, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(32)),
dp1(torch::nn::Dropout(0.1f)),
cv2(torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)),
bn2(torch::nn::BatchNorm2d(64)),
dp2(torch::nn::Dropout(0.2f)),
cv3(torch::nn::Conv2dOptions(64, 64, 3).padding(1).bias(false)),
bn3(torch::nn::BatchNorm2d(64)),
dp3(torch::nn::Dropout(0.2f)),
cv4(torch::nn::Conv2dOptions(64, 64, 3).padding(1).bias(false)),
bn4(torch::nn::BatchNorm2d(64)),
dp4(torch::nn::Dropout(0.2f)),
at_cv3(torch::nn::Conv2dOptions(64, 2, 1).padding(0).bias(false)),
at_bn3(torch::nn::BatchNorm2d(2)),
at_fc1(torch::nn::Linear(2 * largeSize, totSize + 1)),
v_cv3(torch::nn::Conv2dOptions(64, 1, 1).padding(0).bias(false)),
v_bn3(torch::nn::BatchNorm2d(1)),
v_fc1(torch::nn::Linear(largeSize, 64)),
v_fc2(torch::nn::Linear(64, 1)),
device(use_gpu ? torch::kCUDA : torch::kCPU){

	cv1->to(device);
	bn1->to(device);
	dp1->to(device);
	cv2->to(device);
	bn2->to(device);
	dp2->to(device);
	cv3->to(device);
	bn3->to(device);
	dp3->to(device);
	cv4->to(device);
	bn4->to(device);
	dp4->to(device);
	at_cv3->to(device);
	at_bn3->to(device);
	at_fc1->to(device);
	v_cv3->to(device);
	v_bn3->to(device);
	v_fc1->to(device);
	v_fc2->to(device);
	
	register_module("cv1", cv1);
	register_module("bn1", bn1);
	register_module("dp1", dp1);
	register_module("cv2", cv2);
	register_module("bn2", bn2);
	register_module("dp2", dp2);
	register_module("cv3", cv3);
	register_module("bn3", bn3);
	register_module("dp3", dp3);
	register_module("cv4", cv4);
	register_module("bn4", bn4);
	register_module("dp4", dp4);
	register_module("at_cv3", at_cv3);
	register_module("at_bn3", at_bn3);
	register_module("at_fc1", at_fc1);
	register_module("v_cv3", v_cv3);
	register_module("v_bn3", v_bn3);
	register_module("v_fc1", v_fc1);
	register_module("v_fc2", v_fc2);
}

std::pair<torch::Tensor, torch::Tensor> NetImpl::forward(const torch::Tensor& state)
{

	torch::Tensor x = dp1(torch::relu(bn1(cv1(state))));
	x = dp2(torch::relu(bn2(cv2(x))));
	x = dp3(torch::relu(bn3(cv3(x))));
	x = dp4(torch::relu(bn4(cv4(x))));

	torch::Tensor act = torch::relu(at_bn3(at_cv3(x)));
	act = act.view({ -1, 2 * largeSize });
	act = torch::log_softmax(at_fc1(act), 1);

	torch::Tensor val = torch::relu(v_bn3(v_cv3(x)));
	val = val.view({-1, largeSize});
	val = torch::relu(v_fc1(val));
	val = torch::tanh(v_fc2(val));

	return {act, val};
}

PolicyValueNet::PolicyValueNet(const string& model_file, bool use_gpu): use_gpu(use_gpu), policy_value_net(use_gpu)
{
	optimizer = new torch::optim::Adam(policy_value_net->parameters(), l2_const);
	if (model_file != "") {
		torch::load(policy_value_net, model_file);
		cout << "model_loaded" << endl;
		return;
	}
	return;
}

PolicyValueNet& PolicyValueNet::operator=(const PolicyValueNet& other)
{
	if (this == &other)
		return *this;
	this->use_gpu = other.use_gpu;
	this->l2_const = other.l2_const;
	cout << "pv" << endl;
	this->policy_value_net = other.policy_value_net;
	cout << "operator = " << endl;
	delete optimizer;
	cout << "done " << endl;
	this->optimizer = new torch::optim::Adam(this->policy_value_net->parameters(), this->l2_const);

	return *this;
}

pair<array<float, batchSize * (totSize + 1)>, array<float, batchSize> > PolicyValueNet::policy_value(array<float, 5 * batchSize * largeSize>* state_batch)
{
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor current_state = torch::from_blob((*state_batch).data(), { batchSize, 5, boardSize + 2, boardSize + 2 }, options).to(policy_value_net->device);
	pair<torch::Tensor, torch::Tensor> res;

	if (use_gpu) {
		auto r = policy_value_net->forward(current_state);
		res.first = r.first.to(torch::kCPU);
		res.second = r.second.to(torch::kCPU);
	}
	else {
		res = policy_value_net->forward(current_state);
	}

	array<float, batchSize * (totSize + 1)> policy_r;
	array<float, batchSize> value_r;
	float* pt = res.first.data<float>();

	for (int i = 0; i < batchSize * (totSize + 1); ++i)
		policy_r[i] = (*pt++);

	pt = res.second.data<float>();
	for (int i = 0; i < batchSize; ++i)
		value_r[i] = (*pt++);

	return {policy_r, value_r};
}

pair<array<float, totSize + 1>, float> PolicyValueNet::policy_value_fn(const GameManager& game_manager)
{
	const std::vector<int> available = game_manager.get_available();
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor current_state = torch::from_blob(game_manager.current_state().data(), { 1, 5, boardSize + 2, boardSize + 2 }, options).to(policy_value_net->device);
	pair<torch::Tensor, torch::Tensor> res;
	if (use_gpu) {
		auto r = policy_value_net->forward(current_state);
		res.first = r.first.to(torch::kCPU);
		res.second = r.second.to(torch::kCPU);
	}
	else {
		res = policy_value_net->forward(current_state);
	}

	float* pt = res.first.data<float>();
	
	array<float, totSize + 1> fr;
	int temp = 0;
	fr.fill(2.0f);

	for (int i : available) {
		pt += i - temp;
		temp = i;
		fr[i] = exp(*pt);
	}

	return { fr, res.second.index({0, 0}).item<float>() };
}

void PolicyValueNet::train_step(array<float, batchSize * 5 * largeSize>& state_batch, array<float, batchSize * (totSize + 1)>& mcts_probs,
	array<float, batchSize>& winner_batch, float lr) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor sb = torch::from_blob(state_batch.data(), { batchSize, 5, boardSize + 2, boardSize + 2 }, options).to(policy_value_net->device);
	torch::Tensor mp = torch::from_blob(mcts_probs.data(), { batchSize, totSize + 1 }, options).to(policy_value_net->device);
	torch::Tensor wb = torch::from_blob(winner_batch.data(), { batchSize }, options).to(policy_value_net->device);

	optimizer->zero_grad();
	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
	pair<torch::Tensor, torch::Tensor> r = policy_value_net->forward(sb);

	/*int idx = 0;
	cout << "state batch :" << endl;
	for (int i = 0; i < batchSize; ++i) {
		for (int j = 0; j < 5; ++j) {
			for (int k = 0; k < largeSize; ++k) {
				cout << state_batch[idx++] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	idx = 0;
	cout << " mcts_probs : " << endl;
	for (int i = 0; i < batchSize; ++i) {
		for (int j = 0; j < totSize + 1; ++j) {
			cout << mcts_probs[idx++] << " ";
		}
		cout << endl;
	}
	idx = 0;
	cout << "winner :" << endl;
	for (int i = 0; i < batchSize; ++i) {
		cout << winner_batch[i];
		cout << endl;
	}*/

	torch::Tensor value_loss = torch::mse_loss(r.second.view(-1), wb);
	torch::Tensor policy_loss = -torch::mean(torch::sum(mp * r.first, 1));
	torch::Tensor loss = value_loss + policy_loss;
	loss.backward();
	optimizer->step();

	/*torch::Tensor entropy = -torch::mean(torch::sum(torch::exp(r.first) * r.first, 1));
	cout << "loss : " << loss.data() << endl;
	cout << "entropy : " << entropy.data() << endl;*/
	//cout << "train step" << endl;
	return;
}

void PolicyValueNet::save_model(const string& model_file) const
{
	torch::save(policy_value_net, model_file);
}

void PolicyValueNet::load_model(const string& model_file){
	torch::load(policy_value_net, model_file);
}
