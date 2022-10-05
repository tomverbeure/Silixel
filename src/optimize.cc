

#include <map>
#include <unordered_map>
#include <algorithm>

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

    vector<t_lut>                           perm_luts(luts.size());
    std::vector<pair<std::string, int> >    perm_outbits(outbits.size());
    vector<int>                             perm_ones(ones.size());

    if (0) {
        int lut_idx = 0;
        for(auto l: luts){
            printf("lut %d: cfg = %7d,    %d, %d, %d, %d\n", lut_idx, l.cfg, l.inputs[0],l.inputs[1],l.inputs[2],l.inputs[3]);
            ++lut_idx;
        }
        printf("\n");
    }

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

    for(size_type i=0; i != outbits.size(); ++i){
       bool is_ff = outbits[i].second & 1;
       int  node  =  outbits[i].second >> 1;

        perm_outbits[i].first  = outbits[i].first;
        perm_outbits[i].second = (perm[node] << 1) | is_ff;
    }

    for(size_type i=0; i != ones.size(); ++i){
        perm_ones[i] = perm[ones[i]];
    }

    for(size_type i=0; i != luts.size(); ++i){
        luts[i] = perm_luts[i];
    }

    for(size_type i=0; i != outbits.size(); ++i){
        outbits[i] = perm_outbits[i];
    }

    for(size_type i=0; i != ones.size(); ++i){
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



