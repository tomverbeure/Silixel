module memory(clock, out_leds);

    input clock;
    output [7:0] out_leds;

    input latch_en, latch_d;
    output reg latch_q;

    reg [7:0] mem [0:255];

    integer i;
    initial begin
        for(i=0;i<256;i=i+1) begin
            mem[i] = 255-i;
        end
    end


    reg [8:0] cntr = 256;
    always @(posedge clock)
    begin
        if (cntr == 255 || cntr == 511) begin
            cntr <= 0;
        end
        else begin
            cntr <= cntr + 1;

            if (cntr >= 256) begin
                mem[cntr[7:0]] = 255-cntr;
            end
        end
    end

    reg [7:0] mem_rd_data;

    always @(posedge clock) begin
        mem_rd_data = mem[cntr];
    end

    assign out_leds = mem_rd_data;

//    always @(*) begin
//        if (latch_en) begin
//            latch_q <= latch_d;
//        end
//    end

endmodule
