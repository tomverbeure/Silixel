
module tb_memory;

    initial begin
        $dumpfile("waves.vcd");
        $dumpvars;
    end

    reg clock = 0;

    always @(*) begin
        #50;
        clock   <= ~clock;
    end

    initial begin
        repeat(500) begin
            @(posedge clock);
        end 
        $finish;
    end

    wire [7:0] out_leds;
    
    memory u_mem(clock, out_leds);

endmodule
