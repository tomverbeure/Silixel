

#include <map>
#include <algorithm>

#include "profile.h"

void profileHistogram(const vector<t_lut>& luts)
{
    // For each lut, keep track to which other luts the output goes.
    // Use a map so that we can build up the structed out-of-order.
    map<int,vector<int>> luts_fanouts;

    for(int lid = 0; lid < luts.size(); ++lid){
        for(int i=0; i<4; ++i){
            int input_id = luts[lid].inputs[i];

            if (luts_fanouts.find(input_id) == luts_fanouts.end() ){
                luts_fanouts[input_id].push_back(lid);
            }
            else{
                vector<int> &v = luts_fanouts[input_id];

                // The same LUT output might go multiple times to the same LUT. Only add it
                // if the destination LUT isn't already part of the fanout.
                if (std::find(v.begin(), v.end(), lid) == v.end()){
                    v = luts_fanouts[input_id];
                }
            }
        }
    }

    // Now make a histogram of the fanout of each LUT.
    map<int,int> histogram;

    int cnt = 0;
    int max_fanout = 0;
    for(auto lf : luts_fanouts){
#if 0
        printf("src LUT %d: fanout = %d\n", lf.first, (int)lf.second.size());

        if (cnt== 10){
            exit(0);
        }
        ++cnt;
#endif

        int fanout = lf.second.size();
        max_fanout = max(max_fanout, fanout);
        histogram[max_fanout] += 1;
    }

    for(int i=1; i<=max_fanout;++i){
        if (histogram.find(i) != histogram.end()){
            printf("fanout %d: %d LUTs\n", i, histogram[i]);
        }
    }
}

void profileInputs(vector<t_lut>& luts)
{
    map<int,bool> lut_has_ff;

    int nr_ffs = 0;

    for(int lid = 0; lid < luts.size(); ++lid){
        for(int i=0; i<4; ++i){
            int input_id = luts[lid].inputs[i];

            int src_lut_id  = input_id >> 1;
            int src_is_ff   = input_id & 1;

            if (src_is_ff){
                if (lut_has_ff.find(src_lut_id) == lut_has_ff.end() ){
                    lut_has_ff[src_lut_id] = true;
                    // This should be moved to the analyze step...
                    if (src_lut_id < luts.size()){
                        luts[src_lut_id].is_ff = true;
                    }
                    ++nr_ffs;
                }
            }
        }
    }

    printf("total nr luts: %zu, luts with FF: %zu\n", luts.size(), lut_has_ff.size());
}
