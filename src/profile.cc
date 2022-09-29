

#include <map>
#include <unordered_map>
#include <algorithm>
#include <math.h>

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

void profileFFs(vector<t_lut>& luts)
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

double lut_distance(t_lut& lut1, t_lut& lut2)
{
    double distance = 0;

    for(int li1=0;li1<4;++li1){
        if (lut1.inputs[li1] == -1)
            continue;

        long int min_depth = std::numeric_limits<int>::max();
        for(int li2=0; li2<4;++li2){
            if (lut2.inputs[li2] == -1)
                continue;

            min_depth = min(labs( (lut1.inputs[li1]>>1) - (lut2.inputs[li2]>>1) ), min_depth);
        }

        if (min_depth != std::numeric_limits<int>::max()){
            distance += (min_depth * min_depth);
        }
    }

    //printf("distance = %f\n", l, sqrt(distance));

    return sqrt(distance);
}

void profileLocality(vector<t_lut>& luts, int start_lut, int end_lut)
{
    t_lut& prev_lut = luts[start_lut];

    double total_distance = 0;

    for(int i=0;i<4;++i){
        printf("Lut %d inputs: %d\n", start_lut, prev_lut.inputs[i] >> 1);
    }

    // start_lut and end_lut are the boundaries of LUTs at the same depth.
    for(int l=start_lut+1; l<=end_lut; ++l){
        t_lut& cur_lut = luts[l];

        double distance = lut_distance(prev_lut, cur_lut);
#if 0
        for(int pi=0;pi<4;++pi){           // pi for previous lut input
            if (prev_lut_inputs[pi] == -1)
                continue;

            long int min_depth = std::numeric_limits<int>::max();
            for(int ci=0; ci<4;++ci){
                int in = luts[l].inputs[ci]; 
                if (in == -1)
                    continue;
                in >>= 1;

                min_depth = min(labs(prev_lut_inputs[pi]-in), min_depth);
            }
            distance += (min_depth * min_depth);
        }
#endif
        printf("lut %d: distance = %f\n", l, sqrt(distance));
        total_distance += sqrt(distance);

        for(int i=0;i<4;++i){
            prev_lut = cur_lut;
            printf("Lut %d inputs: %d\n", l, prev_lut.inputs[i] >> 1);
        }
    }

    printf("total distance = %f\n", total_distance);
    printf("avg distance = %f\n", total_distance/(end_lut-start_lut));
}

typedef unordered_map<int, unordered_map<int, bool> > t_fanout_map;

int find_closest_lut(vector<t_lut>& luts, t_fanout_map& fanout_map, int ref_lut_idx)
{
    t_lut& ref_lut = luts[ref_lut_idx];

    vector<unordered_map<int, bool>*> closest_luts;

    int min_luts = std::numeric_limits<int>::max();
    int min_luts_idx = 0;
    for(int ri=0; ri<4; ++ri){
        if (ref_lut.inputs[ri] == -1)
            continue;

        unordered_map<int, bool>& fanout_luts = fanout_map[ref_lut.inputs[ri] >> 1];
        if (min_luts <= fanout_map.size()){
            min_luts        = fanout_map.size();
            min_luts_idx    = ri;
        }

        closest_luts.push_back(&fanout_luts);
    }

    for(int cl=0; cl<closest_luts.size();++cl){
        
    }

    return 0;
}


void profileOptimizeLUTs(vector<t_lut>& luts, int start_lut, int end_lut)
{
    // Given a LUT id, we need to come up with a list of LUTs that have an input that is closeby.
    // We could do this, we create a fanout table, where each input is a range of LUT ids (say, the range is 32)
    // and the output is a list of LUTs.

    t_fanout_map fanout_map;

    const int bucket_range = 32;

    for(int lut_idx=start_lut; lut_idx<=end_lut; ++lut_idx){
        for(int in_idx=0; in_idx<4; ++in_idx){
            int in_lut = luts[lut_idx].inputs[in_idx]; 
            if (in_lut < 0)
                continue;
            in_lut >>= 1;

            int in_lut_range = in_lut / bucket_range;
            fanout_map[in_lut_range][lut_idx] = true;
        }
    }


}

