
DESIGN                  ?= silice_vga_demo
LOUVAIN_MAX_FANOUT      ?= 256
LOUVAIN_LEVEL           ?= 3
CUTHILL_MAX_FANOUT      ?= 16

SIM_EXE         = ./build/src/silixel_cuda
#SIM_EXE         = ./build_debug/src/silixel_cuda
LOUVAIN_DIR     = ./louvain-generic/

DESIGN_DIR      = $(DESIGN).design

direct: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) $<

cuthill: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) -c -f $(CUTHILL_MAX_FANOUT) $< 

random: ./$(DESIGN_DIR)/$(DESIGN).blif
	$(SIM_EXE) -r $< 

#============================================================
# Louvain
#============================================================

louvain: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).l$(LOUVAIN_LEVEL).group
	@echo
	@echo "-------------------- Simulating after Louvain reordering"
	@echo
	$(SIM_EXE) -o $< $(DESIGN_DIR)/$(DESIGN).blif

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).l$(LOUVAIN_LEVEL).group: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).graph.tree
	@echo
	@echo "-------------------- Extracting Louvain groups"
	@echo
	$(LOUVAIN_DIR)/hierarchy $< -l $(LOUVAIN_LEVEL) > $@

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).graph.tree: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.bin
	@echo
	@echo "-------------------- Running Louvain algorithm"
	@echo
	$(LOUVAIN_DIR)/louvain $< -v -l -1 > $@

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.bin: $(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.txt
	@echo
	@echo "-------------------- Converting Louvain input"
	@echo
	$(LOUVAIN_DIR)/convert -i $< -o $@

$(DESIGN_DIR)/louvain.$(LOUVAIN_MAX_FANOUT).input.txt: ./$(DESIGN_DIR)/$(DESIGN).blif
	@echo
	@echo "-------------------- Create Louvain input"
	@echo
	$(SIM_EXE) -d $@ -f $(LOUVAIN_MAX_FANOUT) $< 

#============================================================
# Synthesis
#============================================================

./$(DESIGN_DIR)/$(DESIGN).blif: ./$(DESIGN_DIR)/$(DESIGN).v
	/usr/bin/time -v yosys -f verilog $^ -s ./synth/synth.yosys -b blif -o $@ -l ./$(DESIGN_DIR)/$(DESIGN).yosys.log -t

./$(DESIGN_DIR)/$(DESIGN).v:./designs/$(DESIGN).v
	mkdir -p $(DESIGN_DIR)
	cp $< $@

