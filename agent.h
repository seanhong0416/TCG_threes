/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <array>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		//conversion operator, allow object to be explicitly or implicitly casted to string
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		//stod => string to double
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0.0125) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size)){
			//test
			//printf("%lu\n",size);
		}
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

class four_tuple_agent : public weight_agent{
public:
	four_tuple_agent(const std::string& args = "") : weight_agent(args),opcode({ 0, 1, 2, 3 }) {
		//test
		//std::cout << "size of weights: " << net.size() << " " << net[0].size() << " " << net[0][0] << std::endl;
		//std::cout << net_index(1,1,1,1) << std::endl;
	}

	virtual void open_episode(const std::string& flag = "") {
		//reset private data members
		episode_boards.clear();
		episode_rewards.clear();
		episode_values.clear();
	}

	virtual void close_episode(const std::string& flag = "") {
		//start training our agent
		episode_rewards.push_back(0);
		episode_values.push_back(0);
		int len = episode_boards.size();
		//printf("episode_boards length = %lu, episode_rewards length = %lu, episode_values length = %lu\n", episode_boards.size(),episode_rewards.size(),episode_values.size());
		/*
		actual episode length = n
		episode_boards size = n
		episode_rewards size = n+1
		episode_values size = n+1
		*/
		for(int i = len-1;i >= 0;i--){
			double update_value =  alpha * (episode_values[i+1] + episode_rewards[i+1] - episode_values[i]);
			//update_net
			//printf("update value before enter function = %lf\n",update_value);
			update_net(episode_boards[i], update_value);
			//printf("reward = %d, state value = %d, i = %d\n",episode_rewards[i+1], episode_values[i+1], i);
			/*
			printf("=====board being update=====\n");
			for(int j=0;j<4;j++){
				for(int k=0;k<4;k++){
					printf("%d ", episode_boards[i][j][k]);
				}
				printf("\n");
			}
			*/
		}
	}

	void update_net(board& b, double update_value){
		int index_base, index;
		//printf("update value : %lf\n", update_value);

		for(int i=0;i<4;i++){
			index_base = 4*i;
			//update the tuple at row i
			index = net_index(b(index_base), b(index_base+1), b(index_base+2), b(index_base+3));
			net[i][index] += update_value;
			//printf("updating %d %d, value = %lf\n",i,index,net[i][index]);

			//update the tuple at column i
			index = net_index(b(i+0), b(i+4), b(i+8), b(i+12));
			net[i+4][index] += update_value;
		}
	}

	virtual action take_action(const board& b) { 
		board::reward best_reward = -1;
		board::reward reward;
		int best_action = -1;
		double best_after_state_value = -1000;
		board best_after;
		board after;

		for(int op:opcode){
			after = b;
			reward = after.slide(op);
			if(reward == -1) continue;
			double after_state_value = calculate_state_value(after) + reward;
			/*
			printf("=====board before=====\n");
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					printf("%d ", b[i][j]);
				}
				printf("\n");
			}
			
			
			printf("=====board after=====\n");
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					printf("%d ", after[i][j]);
				}
				printf("\n");
			}
			*/
			if(after_state_value > best_after_state_value || best_reward == -1){
				//To compare
				best_after_state_value = after_state_value;
				//To return the action we choose
				best_action = op;
				//To store in the episode_rewards vector
				best_reward = reward;
				//To store in the episode_boards vector
				best_after = after;
			}
			//printf("op:%d\n",op);
			//printf("reward:%d, after_state_value:%lf\n", reward, after_state_value);
			//if(reward!=after_state_value) printf("bingo!\n");
		}
		if(best_reward != -1){
			//store the after board and reward into our vector sothat
			episode_boards.push_back(best_after);
			episode_rewards.push_back(best_reward);
			episode_values.push_back(best_after_state_value);
			//printf("return op = %d\n",best_action);
			return action::slide(best_action);
		}
		else return action();
	}

	double calculate_state_value(const board& b){
		double state_value = 0;
		int index_base, index;

		for(int i=0;i<4;i++){
			index_base = 4*i;
			//plus the tuple at row i
			index = net_index(b(index_base), b(index_base+1), b(index_base+2), b(index_base+3));
			state_value += net[i][index];
			//plus the tuple at column i
			index = net_index(b(i+0), b(i+4), b(i+8), b(i+12));
			state_value += net[4+i][index];
		}

		return state_value;
	}

	int net_index(int index0, int index1, int index2, int index3){
		return index0 | (index1 << 4) | (index2 << 8) | (index3 << 12); 
	}

private:
	std::vector<board> episode_boards;
	std::vector<int> episode_values;
	std::vector<int> episode_rewards;
	std::array<int, 4> opcode;

};

class six_tuple_agent : public weight_agent{
public:
	six_tuple_agent(const std::string& args = "") : weight_agent(args),opcode({ 0, 1, 2, 3 }) {
		//test
		//std::cout << "size of weights: " << net.size() << " " << net[0].size() << " " << net[0][0] << std::endl;
		//std::cout << net_index(1,1,1,1) << std::endl;
		alpha = 0.1/32;
		//initialize tuple index
		tuple_index[0] = {0,1,2,3,4,5};
		tuple_index[1] = {4,5,6,7,8,9};
		tuple_index[2] = {5,6,7,9,10,11};
		tuple_index[3] = {9,10,11,13,14,15};
		//printf("initialization done\n");
	}

	virtual void open_episode(const std::string& flag = "") {
		//reset private data members
		episode_boards.clear();
		episode_rewards.clear();
		episode_values.clear();
	}

	virtual void close_episode(const std::string& flag = "") {
		//start training our agent
		episode_rewards.push_back(0);
		episode_values.push_back(0);
		int len = episode_boards.size();
		//printf("episode_boards length = %lu, episode_rewards length = %lu, episode_values length = %lu\n", episode_boards.size(),episode_rewards.size(),episode_values.size());
		/*
		actual episode length = n
		episode_boards size = n
		episode_rewards size = n+1
		episode_values size = n+1
		*/
		for(int i = len-1;i >= 0;i--){
			double update_value =  alpha * (episode_values[i+1] + episode_rewards[i+1] - episode_values[i]);
			//update_net
			//printf("update value before enter function = %lf\n",update_value);
			update_net(episode_boards[i], update_value);
			//printf("reward = %d, state value = %d, i = %d\n",episode_rewards[i+1], episode_values[i+1], i);
			/*
			printf("=====board being update=====\n");
			for(int j=0;j<4;j++){
				for(int k=0;k<4;k++){
					printf("%d ", episode_boards[i][j][k]);
				}
				printf("\n");
			}
			*/
		}
	}

	void update_net(board& b, double update_value){
		//printf("update value : %lf\n", update_value);

		board  fb = b;

		for(int i=0;i<4;i++){
			int index = 0;
			for(int j=0;j<6;j++){
				index |= fb(tuple_index[i][j]) << (4*j);
			}

			net[i][index] += update_value;
		}

		for(int k=1;k<=3;k++){
			fb.rotate_clockwise();

			for(int i=0;i<4;i++){
				int index = 0;
				for(int j=0;j<6;j++){
					index |= fb(tuple_index[i][j]) << (4*j);
				}

				net[k*4+i][index] += update_value;
			}
		}

		fb = b;
		fb.reflect_horizontal();
		
		for(int i=0;i<4;i++){
			int index = 0;
			for(int j=0;j<6;j++){
				index |= fb(tuple_index[i][j]) << (4*j);
			}

			net[16+i][index] += update_value;
		}

		for(int k=1;k<=3;k++){
			fb.rotate_clockwise();

			for(int i=0;i<4;i++){
				int index = 0;
				for(int j=0;j<6;j++){
					index |= fb(tuple_index[i][j]) << (4*j);
				}

				net[16+k*4+i][index] += update_value;
			}
		}

	}

	virtual action take_action(const board& b) { 
		board::reward best_reward = -1;
		board::reward reward;
		int best_action = -1;
		double best_after_state_value = -1000;
		double best_episode_after_state_value = -1000;
		board best_after;
		board after;

		for(int op:opcode){
			after = b;
			reward = after.slide(op);
			if(reward == -1) continue;
			double episode_after_state_value = calculate_state_value(after);
			double after_state_value = episode_after_state_value + reward;
			/*
			printf("=====board before=====\n");
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					printf("%d ", b[i][j]);
				}
				printf("\n");
			}
			*/
			/*
			printf("=====board after=====\n");
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					printf("%d ", after[i][j]);
				}
				printf("\n");
			}
			after.rotate_clockwise();
			printf("=====after rotation=====\n");
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					printf("%d ", after[i][j]);
				}
				printf("\n");
			}
			*/
			if(after_state_value > best_after_state_value || best_reward == -1){
				//To compare
				best_after_state_value = after_state_value;
				//To return the action we choose
				best_action = op;
				//To store in the episode_rewards vector
				best_reward = reward;
				//To store in the episode_boards vector
				best_after = after;
				//To store in the episode_values vector
				best_episode_after_state_value = episode_after_state_value;
			}
			//printf("op:%d\n",op);
			//printf("reward:%d, after_state_value:%lf\n", reward, after_state_value);
			//if(reward!=after_state_value) printf("bingo!\n");
		}
		if(best_reward != -1){
			//store the after board and reward into our vector sothat
			episode_boards.push_back(best_after);
			episode_rewards.push_back(best_reward);
			episode_values.push_back(best_episode_after_state_value);
			//printf("return op = %d\n",best_action);
			return action::slide(best_action);
		}
		else return action();
	}

	double calculate_state_value(const board& b){
		double state_value = 0;
		board  fb = b;

		for(int i=0;i<4;i++){
			int index = 0;
			for(int j=0;j<6;j++){
				index |= fb(tuple_index[i][j]) << (4*j);
			}

			state_value += net[i][index];
		}

		for(int k=1;k<=3;k++){
			fb.rotate_clockwise();

			for(int i=0;i<4;i++){
				int index = 0;
				for(int j=0;j<6;j++){
					index |= fb(tuple_index[i][j]) << (4*j);
				}

				state_value += net[k*4+i][index];
			}
		}

		fb = b;
		fb.reflect_horizontal();
		
		for(int i=0;i<4;i++){
			int index = 0;
			for(int j=0;j<6;j++){
				index |= fb(tuple_index[i][j]) << (4*j);
			}

			state_value += net[16+i][index];
		}

		for(int k=1;k<=3;k++){
			fb.rotate_clockwise();

			for(int i=0;i<4;i++){
				int index = 0;
				for(int j=0;j<6;j++){
					index |= fb(tuple_index[i][j]) << (4*j);
				}

				state_value += net[16+k*4+i][index];
			}
		}

		return state_value;
	}

	/*
	int net_index(int index0, int index1, int index2, int index3, int index4, int index5){
		return index0 | (index1 << 4) | (index2 << 8) | (index3 << 12) | (index4 << 16) | (index5 << 20); 
	}
	*/

private:
	std::vector<board> episode_boards;
	std::vector<int> episode_values;
	std::vector<int> episode_rewards;
	std::array<int, 4> opcode;
	std::array<std::array<int, 6>,4> tuple_index;

};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

/**
* simple heuristic player
* selects next move based on the score of next state
*/
class heuristic_slider : public agent{
public:
	heuristic_slider(const std::string& args = "") : agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}
	
	virtual action take_action(const board& before) {
		//check
		//std::cout << "took action in heuristic slider" << std::endl;

		board::reward best_reward = -1;
		int best_action = -1;

		for(int op:opcode){
			board::reward reward = board(before).slide(op);
			if(reward > best_reward){
				best_action = op;
				best_reward = reward;
			}
		}

		if(best_reward != -1) return action::slide(best_action);
		else return action();
	}

private:
	std::array<int,4> opcode;

};

class heuristic_slider_kai : public agent{
public:
	heuristic_slider_kai(const std::string& args = "") : agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}
	
	int find_empty_squares(const board& after){
		int empty_square_count = 0;
			for(int i=0;i<16;i++){
				if(after(i) == 0){
					empty_square_count++;
				}
			}
		return empty_square_count;
	}

	int find_monotonic_structure(const board& after, int r, int c){
		//initialization
		int best_len = 0;
		int len = 0;
		monotonic_visited[r][c] = true;
		//basic step
		if(after[r][c] == 0)
			return 0;
		//search up
		if(
			r-1 >= 0 && 
			monotonic_visited[r-1][c] == false && 
			(after[r-1][c] <= after[r][c] || 
			(after[r][c] == 1 && after[r-1][c] == 2))
		){
			len = find_monotonic_structure(after, r-1, c);
			if(len > best_len) best_len = len;
		}
		//search right
		if(
			c+1 < 4 && 
			monotonic_visited[r][c+1] == false &&
			(after[r][c+1] <= after[r][c] || 
			(after[r][c] == 1 && after[r][c+1] == 2))
		){
			len = find_monotonic_structure(after, r, c+1);
			if(len > best_len) best_len = len;
		}
		//search down
		if(
			r+1 < 4 && 
			monotonic_visited[r+1][c] == false &&
			(after[r+1][c] <= after[r][c] || 
			(after[r][c] == 1 && after[r+1][c] == 2))
		){
			len = find_monotonic_structure(after, r+1, c);
			if(len > best_len) best_len = len;
		}
		//search left
		if(
			c-1 >= 0 && 
			monotonic_visited[r][c-1] == false &&
			(after[r][c-1] <= after[r][c] || 
			(after[r][c] == 1 && after[r][c-1] == 2))
		){
			len = find_monotonic_structure(after, r, c-1);
			if(len > best_len) best_len = len;
		}

		return best_len + 1;
	}

	virtual action take_action(const board& before) {
		//coef
		int empty_square_coef = 5;
		if(meta.find("empty_square_coef") != meta.end())
			empty_square_coef = meta["empty_square_coef"];
		int monotonic_structure_coef = 1;
		if(meta.find("monotonic_structure_coef") != meta.end())
			monotonic_structure_coef = meta["monotonic_structure_coef"];
		
		//variables
		board::reward best_reward = -1;
		int best_action = -1;
		int len = 0;
		int best_len = 0;

		for(int op:opcode){
			board after = board(before);
			board::reward reward = after.slide(op);

			//empty squares
			reward += find_empty_squares(after) * empty_square_coef;

			//find longest monotonic structure
			for(int r=0;r<4;r++){
				for(int c=0;c<4;c++){
					//initialize visited array
					for(int i=0;i<4;i++){
						for(int j=0;j<4;j++){
							monotonic_visited[i][j] = 0;
						}
					}
					//find the length of monotonic structure starting from after[r][c]
					len = find_monotonic_structure(after, r, c);
					if(len > best_len) best_len = len;
				}
			}
			reward += best_len * monotonic_structure_coef;

			//find the action with best reward
			if(reward > best_reward){
				best_action = op;
				best_reward = reward;
			}
		}

		if(best_reward != -1) return action::slide(best_action);
		else return action();
	}

private:
	std::array<int,4> opcode;
	std::array<std::array<bool, 4>, 4> monotonic_visited;

};