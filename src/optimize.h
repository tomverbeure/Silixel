#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <vector>
#include <unordered_map>

using namespace std;

#include "read.h"
#include "analyze.h"

void optimizeCuthillMckee(
        vector<t_lut>&                          luts,
        std::vector<pair<std::string, int> >&   outbits,
        vector<int>&                            ones,
        int                                     maxFanout
        );

int optimizeReadGroupFile(const char *filename, unordered_map<int,int>& id2group);
void optimizeSortByGroup(
        vector<t_lut>&                          luts,
        vector<pair<std::string, int> >&        outbits,
        vector<int>&                            ones,
        unordered_map<int, int>                 id2group
        );
void optimizeRandomOrder(
        vector<t_lut>&                          luts,
        vector<pair<std::string, int> >&        outbits,
        vector<int>&                            ones
        );

#endif
