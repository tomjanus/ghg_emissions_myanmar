//
// Created by Jonathan Gomes Selman on 8/6/17.
//

#include "Pareto_Solution.h"

Pareto_Solution::Pareto_Solution(int node_id, const std::vector<std::pair<double, double>> &criteria,
                                 const std::vector<int> &dam_decisions,
                                 const std::vector<Pareto_Solution *> &pareto_decisions) : node_id(node_id),
                                                                                           criteria(criteria),
                                                                                           dam_decisions(dam_decisions),
                                                                                           pareto_decisions(
                                                                                                   pareto_decisions) {
    this->next = nullptr;
    inferior = false;
}

Pareto_Solution::Pareto_Solution(int node_id, const std::pair<double, double> *criteria, const std::vector<int> &dam_decisions,
                                 const std::vector<Pareto_Solution *> &pareto_decisions,
                                 int num_criteria) :  node_id(node_id), dam_decisions(dam_decisions),
                                                      pareto_decisions(pareto_decisions), num_criteria(num_criteria){
    this->next = nullptr;
    inferior = false;
    // Copy in criteria values
    for (int i = 0; i < num_criteria; i++) {
        this->criteria_2[i] = criteria[i];
    }

    // We update the linear preference criteria_lp_2 array when we update the criteria_2 array
}

// Init function considering the linear preference w
// TODO: current only suppport one linear preference
Pareto_Solution::Pareto_Solution(int node_id, const std::pair<double, double> *criteria, const std::vector<int> &dam_decisions,
                                 const std::vector<Pareto_Solution *> &pareto_decisions,
                                 int num_criteria, const std::vector<std::vector<double> > &w, const double eps) :  node_id(node_id), dam_decisions(dam_decisions),
                                                      pareto_decisions(pareto_decisions), num_criteria(num_criteria){
    this->next = nullptr;
    inferior = false;
    // Copy in criteria values
    for (int i = 0; i < num_criteria; i++) {
        this->criteria_2[i] = criteria[i];
    }

    // We update the linear preference criteria_lp_2 array when we update the criteria_2 array
    this->update_linear_preference(0, w, eps);
    //this->criteria_lp_2[0] = criteria[0];
    //double dot_product1 = 0, dot_product2 = 0;
    //for (int i = 1; i < num_criteria; ++i) {
    //    dot_product1 += w[i - 1] * criteria[i].first;
    //    dot_product2 += w[i - 1] * criteria[i].second;
    //}
    //this->criteria_lp_2[1] = std::make_pair(dot_product1, dot_product2);
}

Pareto_Solution::Pareto_Solution(const Pareto_Solution &src) {
    this->next = nullptr;
    this->inferior = src.inferior;
    this->num_criteria = src.num_criteria;
    // Copy in criteria values
    for (int i = 0; i < num_criteria; i++) {
        this->criteria_2[i] = src.criteria_2[i];
    }
    this->node_id = src.node_id;
    this->dam_decisions = src.dam_decisions;
    this->pareto_decisions = src.pareto_decisions;

    // For linear preference. copy construct 
    this->num_criteria_lp = src.num_criteria_lp;
    for (int i = 0; i < num_criteria_lp; ++i) {
        this->criteria_lp_2[i] = src.criteria_lp_2[i];
    }
}

// TODO: Not implement new function for linear preference potential bug
int Pareto_Solution::is_dominant(Pareto_Solution *compare) {
    bool dominate = true; // Flag to see if the current solution dominates the compared solution
    bool is_dominated = true; // Flag to see if the current solution is dominated by compare
    for (int i = 0; i < this->criteria.size(); i++) {
        // Check to see if current solution dominates
        if (this->criteria[i].second > compare->criteria[i].second) { // We want to check some double equality issues!
            is_dominated = false;
        } else if (this->criteria[i].second < compare->criteria[i].second) {
            dominate = false;
        }
    }

    if (dominate && is_dominated) { // If both dominating and dominated then they are the same
        return 2;
    } else if(!dominate && !is_dominated) { // Neither dominates
        return 0;
    } else if (dominate) { // Current solution dominates
        return 1;
    } else { // Current solution is dominated
        return -1;
    }
}

// TODO: Not implement new function for linear preference potential bug
int Pareto_Solution::is_dominant_2(Pareto_Solution *compare) {
    bool dominate = true; // Flag to see if the current solution dominates the compared solution
    bool is_dominated = true; // Flag to see if the current solution is dominated by compare
    for (int i = 0; i < num_criteria; i++) {
        // Check to see if current solution dominates
        if (this->criteria_2[i].second > compare->criteria_2[i].second) { // We want to check some double equality issues!
            is_dominated = false;
        } else if (this->criteria_2[i].second < compare->criteria_2[i].second) {
            dominate = false;
        }
    }

    if (dominate && is_dominated) { // If is both dominating and dominated then they are the same
        return 2;
    } else if(!dominate && !is_dominated) { // Neither dominates
        return 0;
    } else if (dominate) { // Current solution dominates
        return 1;
    } else { // Current solution is dominated
        return -1;
    }
}

int Pareto_Solution::is_dominant_strict(Pareto_Solution *compare) {
    bool dominant = true;
    bool equals = true;
    bool is_dominated = true;
    bool equal_in_one = false;
    // Check for strict domination
    for (int i = 0; i < num_criteria; i++) {
        if (this->criteria_2[i].second <= compare->criteria_2[i].second) {
            dominant = false;
        }

        if (this->criteria_2[i].second >= compare->criteria_2[i].second) {
            is_dominated = false;
        }
        // Check for equality
        if (this->criteria_2[i].second != compare->criteria_2[i].second) {
            equals = false;
        } else {
            equal_in_one = true;
        }
    }

    if (equals) {
        return 2;
    } else if (equal_in_one) {
        return 3;
    } else if (!dominant && !is_dominated) {
        return 0;
    } else if (dominant) {
        return 1;
    } else {
        return -1;
    }
}

int Pareto_Solution::is_dominant_strong(Pareto_Solution *compare) {
    bool equals = true;
    for (int i = 0; i < num_criteria; i++) {
        if (this->criteria_2[i].second != compare->criteria_2[i].second) {
            equals = false;
        }
    }
    if (equals) {
        return 2;
    }

    bool dominant = true;
    bool is_dominated = true;
    for (int i = 0; i < num_criteria; i++) {
        if (this->criteria_2[i].second == compare->criteria_2[i].second) {
            // Check the true values to break ties in random
            // Check to see if current solution dominates
            if (this->criteria_2[i].first > compare->criteria_2[i].first) {
                is_dominated = false;
            } else if (this->criteria_2[i].first < compare->criteria_2[i].first) {
                dominant = false;
            }
        } else if (this->criteria_2[i].second < compare->criteria_2[i].second) {
            dominant = false;
        } else if (this->criteria_2[i].second > compare->criteria_2[i].second) {
            is_dominated = false;
        }
    }

    if (!dominant && !is_dominated) {
        return 0;
    } else if (dominant) {
        return 1;
    } else {
        return -1;
    }
}

int Pareto_Solution::is_dominant_true(Pareto_Solution *compare) {
    bool dominate = true; // Flag to see if the current solution dominates the compared solution
    bool is_dominated = true; // Flag to see if the current solution is dominated by compare
    for (int i = 0; i < num_criteria; i++) {
        // Check to see if current solution dominates
        if (this->criteria_2[i].first > compare->criteria_2[i].first) {
            is_dominated = false;
        } else if (this->criteria_2[i].first < compare->criteria_2[i].first) {
            dominate = false;
        }
    }

    if (dominate && is_dominated) { // If is both dominating and dominated then they are the same
        return 2;
    } else if(!dominate && !is_dominated) { // Neither dominates
        return 0;
    } else if (dominate) {
        return 1;
    } else {
        return -1;
    }
}


bool Pareto_Solution::operator==(const Pareto_Solution &rhs) const {
    // Compare the 'rounded-value' for each criterion
    for (int i = 0; i < num_criteria; i++) {
        if (this->criteria_2[i].second != rhs.criteria_2[i].second) { // Watch out for doubles equals!!
            return false;
        }
    }
    return true;
}

bool Pareto_Solution::operator!=(const Pareto_Solution &rhs) const {
    return !(rhs == *this);
}

int Pareto_Solution::getNum_criteria() const {
    return num_criteria;
}

void Pareto_Solution::update_linear_preference(int num_criteria, const std::vector<std::vector<double> > &w, const double eps) {
    this->criteria_lp_2[0] = this->criteria_2[0];
    double dot_product1, dot_product2; 
    int idx_offset = 1;
    for (int ii = 0; ii < (int)w.size(); ++ii) {
        dot_product1 = dot_product2 = 0.;
        for (int i = 0; i < (int)w[ii].size(); ++i) {
            dot_product1 += w[ii][i] * this->criteria_2[i + idx_offset].first;
            dot_product2 += w[ii][i] * this->criteria_2[i + idx_offset].second;
        }
        idx_offset += (int)w[ii].size();
        // We round the dot_product2 value here
        if (eps != 0) {
            dot_product2 = std::floor(dot_product2 / eps) * eps;
        }
        this->criteria_lp_2[ii + 1] = std::make_pair(dot_product1, dot_product2);
    }
}


