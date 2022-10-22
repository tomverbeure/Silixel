
SIM_EXE     = ./build/src/silixel_cuda

DESIGN      ?= silice_vga_demo

cuthill: ./$(DESIGN)/$(DESIGN).blif
	$(SIM_EXE) -c $< 

direct: ./$(DESIGN)/$(DESIGN).blif
	$(SIM_EXE) $<

./$(DESIGN)/$(DESIGN).blif: ./$(DESIGN)/$(DESIGN).v
	/usr/bin/time -v yosys -f verilog $^ -s ./synth/synth.yosys -b blif -o $@ -l ./$(DESIGN)/$(DESIGN).yosys.log -t

./$(DESIGN)/$(DESIGN).v:./designs/$(DESIGN).v
	mkdir -p $(DESIGN)
	cp $< $@

