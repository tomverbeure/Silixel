#ifndef UTILITY_H
#define UTILITY_H

#include <map>
#include <unordered_map>
#include <algorithm>

#include "optimize.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

void getHighFanoutLuts(
        const vector<t_lut>&                    luts,
        unordered_map<int, int>&                fanout_luts,
        int                                     min_fanout
        );

#endif
