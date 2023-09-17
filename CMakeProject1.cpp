// CMakeProject1.cpp : 애플리케이션의 진입점을 정의합니다.

#include "CMakeProject1.h"
#include "PolicyValue.h"
#include "train.h"
using namespace std;

int main() {
	TrainPipeline* training_pipeline = new TrainPipeline("model3b2300.pt", "model3b2300.pt", true, 2401);
	training_pipeline->run();
	delete training_pipeline;

	/*TrainPipeline::play("model3b2300.pt", false, 2000, 0.1f, true, true);*/
		
	/*GameManager g = GameManager();
	int x, y, v;
	while (true) {
		cin >> x >> y;
		v = g.make_move(x, y, true);
		cout << x << " " << y << " " << v << endl;
		g.switch_turn();
	}*/

	//int x = 1;
	//bool b = x > 0;
	//bool c = x < 0;
	//cout << b << c << endl;
	return 0;
}


