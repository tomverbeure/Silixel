#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <vector>
using namespace std;

#include "read.h"
#include "analyze.h"

void optimizeCache(
        vector<t_lut>&                          luts,
        std::vector<pair<std::string, int> >&   outbits,
        vector<int>&                            ones);

#endif
