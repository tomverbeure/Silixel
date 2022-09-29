#ifndef PROFILE_H
#define PROFILE_H

#include <vector>
using namespace std;

#include "read.h"
#include "analyze.h"

void profileHistogram(const vector<t_lut>& luts);
void profileFFs(vector<t_lut>& luts);
void profileLocality(vector<t_lut>& luts, int start_lut, int end_lut);
void profileOptimizeLUTs(vector<t_lut>& luts, int start_lut, int end_lut);

#endif
