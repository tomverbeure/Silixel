#ifndef PROFILE_H
#define PROFILE_H

#include <vector>
#include <string>

using namespace std;

#include "read.h"
#include "analyze.h"

void profileHistogram(const vector<t_lut>& luts);
void profileBoost(const vector<t_lut>& luts);
void profileInputDifferences(
    const vector<t_lut>&    luts, 
    const vector<int>&      step_starts,
    const vector<int>&      step_ends,
    int level
    );

void profileDumpLouvainGraph(
    const vector<t_lut>&    luts,
    string &                egdesFile,
    int                     max_fanout 
    );

void profileDumpLeidenGraph(
    const vector<t_lut>&    luts
    );

#endif
