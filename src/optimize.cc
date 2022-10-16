

#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdlib>

#include "optimize.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

// TODO:
// 1. Calculate bandwidth
// 2. RMC first, bandwidth 
// 3. Remove high fanout nets from graph after step 2, RMC again, bandwidth
// 4. Remove high fanout nets from original graph, RMC, bandwidth
// 5. Check if BW after step 3 and 4 is similar
void optimizeCache(
        vector<t_lut>&                          luts,
        std::vector<pair<std::string, int> >&   outbits,
        vector<int>&                            ones)
{
    using namespace boost;
    using namespace std;

   cout << "===========================================================" << endl;
   cout << "Optimizing graph..." << endl;
   cout << "===========================================================" << endl;

    typedef adjacency_list< 
            listS,                      // how to store edge list
            vecS,                       // how to store vertex list
            undirectedS,
            property< vertex_color_t, default_color_type,           // Vertex properties
                property< vertex_degree_t, int > > >
        MyGraph;

    typedef graph_traits< MyGraph > GraphTraits;
    typedef graph_traits< MyGraph >::vertex_descriptor Vertex;
    typedef graph_traits< MyGraph >::vertices_size_type size_type;

    MyGraph netlist_graph(luts.size());

    //============================================================
    // Add all LUTs to the netlist graph
    //============================================================
    for(int lut_idx=0; lut_idx<luts.size(); ++lut_idx){
        const t_lut& cur_lut = luts[lut_idx];

        for(int input_idx=0; input_idx<4; ++input_idx){
            if (cur_lut.inputs[input_idx] == -1)
                continue;

            add_edge(lut_idx, cur_lut.inputs[input_idx] >> 1, netlist_graph);
        }
    }

    printf("Num vertices: %zu\n", num_vertices(netlist_graph));
    printf("Graph bandwidth: %zu\n", bandwidth(netlist_graph));

#if 1
    //============================================================
    // Create a graph degree histogram
    //============================================================
    {
        graph_traits< MyGraph >::vertex_iterator                ui, ui_end;
        std::unordered_map<int, int>                            degree_histogram;
        property_map< MyGraph, vertex_degree_t >::type          deg = get(vertex_degree, netlist_graph);
        auto                                                    max_deg_v = *vertices(netlist_graph).first;
        int                                                     max_deg = 0;

        for (boost::tie(ui, ui_end) = vertices(netlist_graph); ui != ui_end; ++ui){
            int cur_deg = degree(*ui, netlist_graph);
            deg[*ui] = cur_deg;
    
            if (cur_deg > max_deg){
                max_deg = cur_deg;
                max_deg_v = *ui;
            }
    
            if (degree_histogram.find(cur_deg) == degree_histogram.end()){
                degree_histogram[cur_deg] = 1;
            }
            else{
                degree_histogram[cur_deg] += 1;
            }
        }
    
        printf("max degree: %d\n", max_deg);
    
        std::map<int, int> degree_histogram_sorted(degree_histogram.begin(), degree_histogram.end());
        for(auto it = degree_histogram_sorted.begin(); it != degree_histogram_sorted.end(); ++it){
            printf("degree %d -> %d entries\n", it->first, it->second);
        }
    }
#endif

    //============================================================
    // Calculate reverse Cuthill-McKee 
    //============================================================

    // index_map converts from a Vertex object to an integer index.
    property_map< MyGraph, vertex_index_t >::type index_map = get(vertex_index, netlist_graph);

    std::vector< Vertex >       inv_perm(num_vertices(netlist_graph));
    std::vector< size_type >    perm(num_vertices(netlist_graph));

    {
        // reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(
                netlist_graph,
                inv_perm.rbegin(), 
                get(vertex_color, netlist_graph), 
                make_degree_map(netlist_graph)
                );

#if 0
        cout << "Reverse Cuthill-McKee ordering (new to old):" << endl;
        cout << "  ";
        for (std::vector< Vertex >::const_iterator i = inv_perm.begin(); i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;
#endif

        // perm[i] contains the new location of LUT[i].
        for (size_type c = 0; c != inv_perm.size(); ++c){
            perm[index_map[inv_perm[c]]] = c;
        }

#if 0
        cout << "Reverse Cuthill-McKee ordering (old to new):" << endl;
        cout << "  ";
        for (auto c: perm)
            cout << c << " ";
        cout << endl;
#endif

        std::cout << "  bandwidth: "
                  << bandwidth(
                            netlist_graph,
                            make_iterator_property_map( &perm[0], index_map, perm[0]))
                  << std::endl;
    }


    if (0) {
        int lut_idx = 0;
        for(auto l: luts){
            printf("lut %d: cfg = %7d,    %d, %d, %d, %d\n", lut_idx, l.cfg, l.inputs[0],l.inputs[1],l.inputs[2],l.inputs[3]);
            ++lut_idx;
        }
        printf("\n");
    }

    //============================================================
    // Shuffle LUTs around, as calculated by Cuthill-McKee
    //============================================================

    // Instead of in-place shuffling, do a copy into a new LUTs vector, because
    // I don't quite understand the memory model...
    vector<t_lut>                           perm_luts(luts.size());
    std::vector<pair<std::string, int> >    perm_outbits(outbits.size());
    vector<int>                             perm_ones(ones.size());

    for(size_type i=0; i != perm.size(); ++i){
        perm_luts[perm[i]].cfg    = luts[i].cfg;
        for(int in=0; in<4; ++in){
            if (luts[i].inputs[in] != -1){
                bool is_ff = luts[i].inputs[in] & 1;
                int  node  = luts[i].inputs[in] >> 1;
                perm_luts[perm[i]].inputs[in]    = (perm[node] << 1) | is_ff;
            }
            else
                perm_luts[perm[i]].inputs[in]    = -1;
        }
    }

    for(size_t i=0; i != outbits.size(); ++i){
        bool is_ff = outbits[i].second & 1;
        int  node  =  outbits[i].second >> 1;

        perm_outbits[i].first  = outbits[i].first;
        perm_outbits[i].second = (perm[node] << 1) | is_ff;
    }

    for(size_t i=0; i != ones.size(); ++i){
        bool is_ff = ones[i] & 1;
        int  node  =  ones[i] >> 1;

        perm_ones[i] = perm[node] << 1 | is_ff;
    }

    //  Copy everything back to the original vectors.
    for(size_t i=0; i != luts.size(); ++i){
        luts[i] = perm_luts[i];
    }

    for(size_t i=0; i != outbits.size(); ++i){
        outbits[i] = perm_outbits[i];
    }

    for(size_t i=0; i != ones.size(); ++i){
        ones[i] = perm_ones[i];
    }

    if (0) {
        int lut_idx = 0;
        for(auto l: luts){
            printf("lut %d: cfg = %7d,    %d, %d, %d, %d\n", lut_idx, l.cfg, l.inputs[0],l.inputs[1],l.inputs[2],l.inputs[3]);
            ++lut_idx;
        }
        printf("\n");
    }

}

int optimizeReadGroupFile(const char *filename, unordered_map<int,int>& id2group)
{
    fprintf(stderr, "Reading LUT id to group mapping file '%s'\n", filename);
    ifstream finput;
    finput.open(filename, fstream::in);
    if (finput.is_open() != true){
        fprintf(stderr, "Can't open group file '%s'. Aborting...\n", filename);
        exit(-1);
    }

    int max_group_nr = -1;
    id2group.clear();

    while(!finput.eof()){
        int lut_id, group_nr;

        finput >> lut_id >> group_nr;

        if (id2group.find(lut_id) == id2group.end()){
            id2group[lut_id] = group_nr;
            max_group_nr = max(max_group_nr, group_nr);
        }
        else{
            fprintf(stderr, "LUT ID %d (group %d) has duplicate entries?! Ignoring...\n", lut_id, group_nr);
        }
    }
    finput.close();

    fprintf(stderr, "%zu LUTs remapped to %d groups.\n", id2group.size(), max_group_nr);

    return max_group_nr+1;
}

void optimizeSortByGroup(
        vector<t_lut>&                          luts,
        vector<pair<std::string, int> >&        outbits,
        vector<int>&                            ones,
        const unordered_map<int, int>           id2group
        )
{
    // Find the number of groups
    int max_group_nr = -1;
    for(auto it=id2group.begin(); it!=id2group.end(); ++it){
        max_group_nr = max(max_group_nr, it->second);
    }

    fprintf(stderr, "%d groups...\n", max_group_nr+1);

    // Now create a list with all the LUTs for each group.
    vector<vector<int>>     groups_with_luts(max_group_nr+1);
    for(auto it=id2group.begin(); it!=id2group.end(); ++it){
        groups_with_luts[it->second].push_back(it->first);
    }

    fprintf(stderr, "1\n");

    // Create permutation table from new position to old position...
    vector<int> inv_perm;
    inv_perm.reserve(luts.size());

    for(auto g_it=groups_with_luts.begin(); g_it!=groups_with_luts.end();++g_it){
        // Sort to maintain order the same order in the group as the one before grouping.
//        sort(g_it->begin(), g_it->end());

        for(auto lut_id_it=g_it->begin(); lut_id_it!=g_it->end(); ++lut_id_it){
            inv_perm.push_back(*lut_id_it);
        }
    }

    // Create permutation table from old position to new position
    vector<int> perm;
    perm.resize(luts.size());

    for(size_t i=0; i<inv_perm.size(); ++i){
        perm[inv_perm[i]] = i;
        //perm[i] = i;
    }

    fprintf(stderr, "2, hello\n");

    vector<t_lut>                           perm_luts(luts.size());
    std::vector<pair<std::string, int> >    perm_outbits(outbits.size());
    vector<int>                             perm_ones(ones.size());

    for(size_t i=0; i != perm.size(); ++i){
        perm_luts[perm[i]].cfg    = luts[i].cfg;
        for(int in=0; in<4; ++in){
            if (luts[i].inputs[in] != -1){
                bool is_ff = luts[i].inputs[in] & 1;
                int  node  = luts[i].inputs[in] >> 1;
                perm_luts[perm[i]].inputs[in]    = (perm[node] << 1) | is_ff;
            }
            else
                perm_luts[perm[i]].inputs[in]    = -1;
        }
    }
    
    fprintf(stderr, "3\n");

    for(size_t i=0; i<outbits.size(); ++i){
        bool is_ff = outbits[i].second & 1;
        int  node  =  outbits[i].second >> 1;

        perm_outbits[i].first  = outbits[i].first;
        perm_outbits[i].second = (perm[node] << 1) | is_ff;
    }

    for(size_t i=0; i<ones.size(); ++i){
        bool is_ff = ones[i] & 1;
        int  node  =  ones[i] >> 1;

        perm_ones[i] = perm[node] << 1 | is_ff;
    }

    //  Copy everything back to the original vectors.
    for(size_t i=0; i<luts.size(); ++i){
        luts[i] = perm_luts[i];
    }

    for(size_t i=0; i<outbits.size(); ++i){
        outbits[i] = perm_outbits[i];
    }

    for(size_t i=0; i<ones.size(); ++i){
        ones[i] = perm_ones[i];
    }
}


void optimizeRandomOrder(
        vector<t_lut>&                          luts,
        vector<pair<std::string, int> >&        outbits,
        vector<int>&                            ones
        )
{
    // Create permutation table from old position to new position
    vector<int> perm;
    perm.resize(luts.size());

    for(size_t i=0; i<perm.size(); ++i){
        perm[i] = i;
    }

    srand(0);
    int rand_max = -1;
    for(int j=0; j<10;++j){
        for(int i=0; i<luts.size(); ++i){
            int random = rand() % luts.size();
            rand_max = max(rand_max, random);
            int old = perm[i];
            perm[i] = perm[random];
            perm[random] = old;
        }
    }


    vector<t_lut>                           perm_luts(luts.size());
    std::vector<pair<std::string, int> >    perm_outbits(outbits.size());
    vector<int>                             perm_ones(ones.size());

    for(size_t i=0; i != perm.size(); ++i){
        perm_luts[perm[i]].cfg    = luts[i].cfg;
        for(int in=0; in<4; ++in){
            if (luts[i].inputs[in] != -1){
                bool is_ff = luts[i].inputs[in] & 1;
                int  node  = luts[i].inputs[in] >> 1;
                perm_luts[perm[i]].inputs[in]    = (perm[node] << 1) | is_ff;
            }
            else
                perm_luts[perm[i]].inputs[in]    = -1;
        }
    }
    
    fprintf(stderr, "3\n");

    for(size_t i=0; i<outbits.size(); ++i){
        bool is_ff = outbits[i].second & 1;
        int  node  =  outbits[i].second >> 1;

        perm_outbits[i].first  = outbits[i].first;
        perm_outbits[i].second = (perm[node] << 1) | is_ff;
    }

    for(size_t i=0; i<ones.size(); ++i){
        bool is_ff = ones[i] & 1;
        int  node  =  ones[i] >> 1;

        perm_ones[i] = perm[node] << 1 | is_ff;
    }

    //  Copy everything back to the original vectors.
    for(size_t i=0; i<luts.size(); ++i){
        luts[i] = perm_luts[i];
    }

    for(size_t i=0; i<outbits.size(); ++i){
        outbits[i] = perm_outbits[i];
    }

    for(size_t i=0; i<ones.size(); ++i){
        ones[i] = perm_ones[i];
    }

    int idem_cnt = 0;
    int no_conn_cnt = 0;
    fprintf(stderr, "Max random: %d, RAND_MAX: %d\n", rand_max, RAND_MAX);
    for(int i=0; i<luts.size(); ++i){
        for(int j=0;j<4;++j){
            if (luts[i].inputs[j]==-1){
                ++no_conn_cnt;
                continue;
            }
#if 0
            printf("%d: %d\n", i, luts[i].inputs[j]>>1);
#endif

            if (i==(luts[i].inputs[j]>>1)){
                ++idem_cnt;
            }
        }
    }
    fprintf(stderr, "idem_cnt: %d\n", idem_cnt);
    fprintf(stderr, "no_conn_cnt: %d\n", no_conn_cnt);

}


