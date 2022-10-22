
DESIGN                  ?= silice_vga_demo
LOUVAIN_MAX_FANOUT      ?= 256
LOUVAIN_LEVEL           ?= 3

SIM_EXE         = ./build/src/silixel_cuda
#SIM_EXE         = ./build_debug/src/silixel_cuda
LOUVAIN_DIR     = ./louvain-generic/

DESIGN_DIR      = $(DESIGN).design

direct: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) $<

cuthill: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) -c $< 

random: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) -r $< 

louvain: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).l$(LOUVAIN_LEVEL).group

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).l$(LOUVAIN_LEVEL).group: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).graph.tree
	$(LOUVAIN_DIR)/hierarchy $< -l $(LOUVAIN_LEVEL) > $@

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).graph.tree: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.bin
	$(LOUVAIN_DIR)/louvain $< -v -l -1 > $@

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.bin: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.txt
	$(LOUVAIN_DIR)/convert -i $< -o $@

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.txt: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) -d $@ -f $(LOUVAIN_MAX_FANOUT) $< 

./$(DESIGN_DIR)/$(DESIGN).blif: ./$(DESIGN_DIR)/$(DESIGN).v
	/usr/bin/time -v yosys -f verilog $^ -s ./synth/synth.yosys -b blif -o $@ -l ./$(DESIGN_DIR)/$(DESIGN).yosys.log -t

./$(DESIGN_DIR)/$(DESIGN).v:./designs/$(DESIGN).v
	mkdir -p $(DESIGN_DIR)
	cp $< $@

