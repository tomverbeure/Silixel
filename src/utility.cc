

#include <map>
#include <unordered_map>
#include <algorithm>

#include "optimize.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

void getHighFanoutLuts(
        const vector<t_lut>&                          luts,
        unordered_map<int, int>&                fanout_luts,
        int                                     min_fanout
        )
{

    // Build structure with all the fanouts of a given LUT
    unordered_map<int, int>     fanout_counts;

    for(int lid=0; lid<luts.size(); ++lid){
        for(int i=0; i<4; ++i){
            int input_id = luts[lid].inputs[i];
            if (input_id == -1)
                continue;
            input_id >>= 1;

            if (fanout_counts.find(input_id) == fanout_counts.end())
                fanout_counts[input_id] = 1;
            else
                fanout_counts[input_id] += 1;
        }
    }

    // Copy over all LUTs with a fanout that's at least min_fanout
    fanout_luts.clear();

    for(auto it=fanout_counts.begin(); it != fanout_counts.end(); ++it){
        if (it->second >= min_fanout)
            fanout_luts[it->first] = it->second;
    }
}

