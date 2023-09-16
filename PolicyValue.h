#pragma once

#include <utility>
#include <torch/torch.h>
#include "GameManager.h"
#include <deque>
#include <string>


const int batchSize = 512;

class NetImpl : public torch::nn::Module{
public:
	NetImpl(bool use_gpu);
	std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state);
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Dropout dp1;
	torch::nn::Conv2d cv2;
	torch::nn::BatchNorm2d bn2;
	torch::nn::Dropout dp2;
	torch::nn::Conv2d cv3;
	torch::nn::BatchNorm2d bn3;
	torch::nn::Dropout dp3;
	torch::nn::Conv2d cv4;
	torch::nn::BatchNorm2d bn4;
	torch::nn::Dropout dp4;
	torch::nn::Conv2d at_cv3;
	torch::nn::BatchNorm2d at_bn3;
	torch::nn::Linear at_fc1;
	torch::nn::Conv2d v_cv3;
	torch::nn::BatchNorm2d v_bn3;
	torch::nn::Linear v_fc1;
	torch::nn::Linear v_fc2;

	torch::Device device;
};

TORCH_MODULE(Net);

class PolicyValueNet {
private:
	bool use_gpu;
	float l2_const = 0.0001f;
	torch::optim::Adam* optimizer;

public:
	Net policy_value_net;

	PolicyValueNet(const std::string& model_file, bool use_gpu);
	PolicyValueNet& operator=(const PolicyValueNet& pv);
	std::pair< std::array<float, batchSize * (totSize + 1)>, std::array<float, batchSize> > policy_value(std::array<float, 5 * batchSize * largeSize>* state_batch);
	std::pair<std::array<float, totSize + 1>, float> policy_value_fn(const GameManager& game_manager);
	void train_step(std::array<float, 5 * batchSize * largeSize>& state_batch, std::array<float, batchSize * (totSize + 1)>& mcts_probs,
		std::array<float, batchSize>& winner_batch, float lr);
	void save_model(const std::string& model_file) const;
	void load_model(const std::string& model_file);
};