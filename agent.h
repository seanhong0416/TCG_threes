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
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
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
		for (size_t size; in >> size; net.emplace_back(size));
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
		opcode({ 0, 1, 2, 3 }) {
			if(meta.find("empty_square_coef") != meta.end())
				empty_square_coef = meta["empty_square_coef"];
			if(meta.find("monotonic_structure_coef") != meta.end())
				monotonic_structure_coef = meta["monotonic_structure_coef"];
			if(meta.find("largest_placement_value_multiplier") != meta.end())
				largest_placement_value_multiplier = meta["largest_placement_value_multiplier"];
			//create the value table for largest tile
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					largest_placement_value[i][j] = 1;
				}
			}
			for(int i=0;i<4;i++){
				largest_placement_value[i][0] += largest_placement_value_multiplier;
				largest_placement_value[i][3] += largest_placement_value_multiplier;
			}
			for(int j=0;j<4;j++){
				largest_placement_value[0][j] += largest_placement_value_multiplier;
				largest_placement_value[0][j] += largest_placement_value_multiplier;
			}
		}
	
	int find_empty_squares(const board& after){
		int empty_square_count = 0;
			for(int i=0;i<16;i++){
				if(after(i) == 0){
					empty_square_count++;
				}
			}
		return empty_square_count;
	}

	int find_monotonic_structure(const board& after, int r, int c, unsigned int last_tile){
		//initialization
		int best_len = 0;
		int len = 0;
		monotonic_visited[r][c] = true;
		//basic step
		//if(last_tile == 0)
		//	return 0;

		//search up
		if(
			r-1 >= 0 && 
			monotonic_visited[r-1][c] == false && 
			(after[r-1][c] <= last_tile || 
			(after[r][c] == 1 && after[r-1][c] == 2))
		){	
			len = find_monotonic_structure(after, r-1, c, (after[r][c]==0)?last_tile:after[r][c]);
			if(len > best_len) best_len = len;
		}
		//search right
		if(
			c+1 < 4 && 
			monotonic_visited[r][c+1] == false &&
			(after[r][c+1] <= last_tile || 
			(after[r][c] == 1 && after[r][c+1] == 2))
		){
			len = find_monotonic_structure(after, r, c+1, (after[r][c]==0)?last_tile:after[r][c]);
			if(len > best_len) best_len = len;
		}
		//search down
		if(
			r+1 < 4 && 
			monotonic_visited[r+1][c] == false &&
			(after[r+1][c] <= last_tile || 
			(after[r][c] == 1 && after[r+1][c] == 2))
		){
			len = find_monotonic_structure(after, r+1, c, (after[r][c]==0)?last_tile:after[r][c]);
			if(len > best_len) best_len = len;
		}
		//search left
		if(
			c-1 >= 0 && 
			monotonic_visited[r][c-1] == false &&
			(after[r][c-1] <= last_tile || 
			(after[r][c] == 1 && after[r][c-1] == 2))
		){
			len = find_monotonic_structure(after, r, c-1, (after[r][c]==0)?last_tile:after[r][c]);
			if(len > best_len) best_len = len;
		}

		//std::cout << after[r][c] << std::endl;
		return best_len + (after[r][c]==0)?0:1;
	}

	virtual action take_action(const board& before) {

		//variables
		board::reward best_reward = -1;
		int best_action = -1;
		int len = 0;
		int best_len = 0;
		int largest_x = 0, largest_y = 0;
		unsigned int largest_value = 0;

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
					len = find_monotonic_structure(after, r, c, after[r][c]);
					if(len > best_len) best_len = len;
				}
			}
			reward += best_len * monotonic_structure_coef;

			//position of largest tile
			//find the position of the largest tile 
			for(int i=0;i<4;i++){
				for(int j=0;j<4;j++){
					if(after[i][j] > largest_value){
						largest_value = after[i][j];
						largest_x = i;
						largest_y = j;
					}
				}
			}
			//plus the position value to our reward using position placement value table
			reward += largest_placement_value[largest_x][largest_y];

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
	std::array<std::array<int, 4>, 4> largest_placement_value;
	int empty_square_coef = 5;
	int monotonic_structure_coef = 1;
	int largest_placement_value_multiplier = 2;
};