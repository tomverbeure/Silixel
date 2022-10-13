

#include <map>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>



#include "profile.h"

void profileHistogram(const vector<t_lut>& luts)
{
    // For each lut, keep track to which other luts the output goes.
    // Use a map so that we can build up the structed out-of-order.
    unordered_map<int,vector<int>> luts_fanouts;

    for(int lid = 0; lid < luts.size(); ++lid){
        for(int i=0; i<4; ++i){
            int input_id = luts[lid].inputs[i];
            if (input_id == -1)
                continue;

            input_id >>= 1;

            if (luts_fanouts.find(input_id) == luts_fanouts.end() ){
                luts_fanouts[input_id].push_back(lid);
            }
            else{
                vector<int> &fanouts = luts_fanouts[input_id];

                // The same LUT output might go multiple times to the same LUT. Only add it
                // if the destination LUT isn't already part of the fanout.
                if (std::find(fanouts.begin(), fanouts.end(), lid) == fanouts.end()){
                    fanouts.push_back(lid);
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
void profileBoost(const vector<t_lut>& luts)
{
    using namespace boost;
    using namespace std;

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

    MyGraph G(luts.size());

    for(int lut_idx=0; lut_idx<luts.size(); ++lut_idx){
        const t_lut& cur_lut = luts[lut_idx];

        for(int input_idx=0; input_idx<4; ++input_idx){
            if (cur_lut.inputs[input_idx] == -1)
                continue;

            add_edge(lut_idx, cur_lut.inputs[input_idx] >> 1, G);
        }
    }

    printf("Num vertices: %zu\n", num_vertices(G));
    printf("Graph bandwidth: %zu\n", bandwidth(G));

    auto v = *vertices(G).first;
    printf("Vertex.first, degree: %zu\n", degree(v, G));
#if 0
    remove_vertex(v, G);
#endif

    printf("Num vertices: %zu\n", num_vertices(G));

    graph_traits< MyGraph >::vertex_iterator ui, ui_end;

    std::unordered_map<int, int> degree_histogram;

    property_map< MyGraph, vertex_degree_t >::type deg = get(vertex_degree, G);
    v = *vertices(G).first;
    int max_deg = 0;
    for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
        int cur_deg = degree(*ui, G);
        deg[*ui] = cur_deg;

        if (cur_deg > max_deg){
            max_deg = cur_deg;
            v = *ui;
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

#if 0
    typedef typename GraphTraits::edge_descriptor edge_descriptor;
    edge_descriptor e;

    auto edge_it = out_edges(v, G);
    for(auto e = edge_it.first; e != edge_it.second; ++e){
        remove_edge(e, G);
    }
#endif
    clear_vertex(v, G);
    printf("Degree after removing edges: %zu\n", degree(v, G));

    for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui){
        if (degree(*ui, G) > 32){
            clear_vertex(*ui, G);
        }
    }

    property_map< MyGraph, vertex_index_t >::type index_map
        = get(vertex_index, G);

    std::vector< Vertex > inv_perm(num_vertices(G));
    std::vector< size_type > perm(num_vertices(G));
    {
        Vertex s = vertex(6, G);
        // reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G),
            get(vertex_degree, G));
        cout << "Reverse Cuthill-McKee ordering starting at: " << s << endl;
#if 0
        cout << "  ";
        for (std::vector< Vertex >::const_iterator i = inv_perm.begin();
             i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;
#endif

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                  << bandwidth(G,
                         make_iterator_property_map(
                             &perm[0], index_map, perm[0]))
                  << std::endl;
    }
    {
        Vertex s = vertex(0, G);
        // reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G),
            get(vertex_degree, G));
        cout << "Reverse Cuthill-McKee ordering starting at: " << s << endl;
#if 0
        cout << "  ";
        for (std::vector< Vertex >::const_iterator i = inv_perm.begin();
             i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;
#endif

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                  << bandwidth(G,
                         make_iterator_property_map(
                             &perm[0], index_map, perm[0]))
                  << std::endl;
    }

    {
        // reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(
            G, inv_perm.rbegin(), get(vertex_color, G), make_degree_map(G));

        cout << "Reverse Cuthill-McKee ordering:" << endl;
#if 0
        cout << "  ";
        for (std::vector< Vertex >::const_iterator i = inv_perm.begin();
             i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;
#endif

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                  << bandwidth(G,
                         make_iterator_property_map(
                             &perm[0], index_map, perm[0]))
                  << std::endl;
    }



    exit(0);
}


// Loop through all LUTs of a level in steps of 32 (size of a warp),
// calculate the distances between inputs siganls after sorting, and 
// create a histogram.
// This is supposed to give an indication of how efficient the GPU should
// be able to fetch the data from DRAM.
void profileInputDifferences(
    const vector<t_lut>&    luts, 
    const vector<int>&      step_starts,
    const vector<int>&      step_ends,
    int level
    )
{
   unordered_map<int, int> warp_deltas;

    for(int lid_warp=step_starts[level]; lid_warp <= step_ends[level]; lid_warp += 32){
         unordered_multiset<int> input_ids;

        // Get all input ids within the same warp
        for(int lid=lid_warp; lid <= min(step_ends[level], lid_warp+31); ++lid){
            for(int i=0; i<4; ++i){
                int input_id = luts[lid].inputs[i];
                if (input_id == -1)
                    continue;

                input_id >>= 1;
                input_ids.insert(input_id);
            }
        }

        vector<int> input_ids_sorted(input_ids.begin(), input_ids.end());
        sort(input_ids_sorted.begin(), input_ids_sorted.end());

        int prev_id = -1;
        for(auto id: input_ids_sorted){
            if (prev_id == -1){
                prev_id = id;
                continue;
            }

            int delta = abs(prev_id-id);

            if (warp_deltas.find(delta) == warp_deltas.end()){
                warp_deltas[delta] = 1;
            }
            else{
                warp_deltas[delta] += 1;
            }
        }
    }

    vector<int> deltas_ordered;
    for(auto w: warp_deltas)
        deltas_ordered.push_back(w.first);
    sort(deltas_ordered.begin(), deltas_ordered.end());

    for(auto d: deltas_ordered){
        printf("delta %d: %d\n", d, warp_deltas[d]);
    }
}


void profileDumpLouvainGraph(
    const vector<t_lut>&    luts
    )
{
    for(int lid=0; lid<luts.size(); ++lid){
        for(int i=0; i<4; ++i){
            int input_id = luts[lid].inputs[i];
            if (input_id == -1)
                continue;
            input_id >>= 1;

            printf("%d %d\n", input_id, lid);
        }
    }

}

void profileDumpLeidenGraph(
    const vector<t_lut>&    luts
    )
{
    unordered_map<long, bool> edge_tags;

    for(int lid=0; lid<luts.size(); ++lid){
        for(int i=0; i<4; ++i){
            int input_id = luts[lid].inputs[i];
            if (input_id == -1)
                continue;
            input_id >>= 1;

            long edge_tag = min(input_id, lid) + max(input_id, lid) * luts.size();
            if (edge_tags.find(edge_tag) == edge_tags.end()){
                printf("%d\t%d\n", input_id, lid);
                edge_tags[edge_tag] = true;
            }
        }
    }

}
